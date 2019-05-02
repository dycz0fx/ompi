#include "coll_future.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_future_trigger.h"

void mac_coll_future_set_allreduce_argu(mca_allreduce_argu_t *argu,
                                        mca_coll_task_t *cur_task,
                                        void *sbuf,
                                        void *rbuf,
                                        int seg_count,
                                        struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        int root_up_rank,
                                        int root_low_rank,
                                        struct ompi_communicator_t *up_comm,
                                        struct ompi_communicator_t *low_comm,
                                        int num_segments,
                                        int cur_seg,
                                        int w_rank,
                                        int last_seg_count,
                                        bool noop,
                                        ompi_request_t *req,
                                        int *completed){
    argu->cur_task = cur_task;
    argu->sbuf = sbuf;
    argu->rbuf = rbuf;
    argu->seg_count = seg_count;
    argu->dtype = dtype;
    argu->op = op;
    argu->root_up_rank = root_up_rank;
    argu->root_low_rank = root_low_rank;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->num_segments = num_segments;
    argu->cur_seg = cur_seg;
    argu->w_rank = w_rank;
    argu->last_seg_count = last_seg_count;
    argu->noop = noop;
    argu->req = req;
    argu->completed = completed;
}

/*
 * Async implementation of allreduce.
 */

int
mca_coll_future_allreduce_intra(const void *sbuf,
                                void *rbuf,
                                int count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module){
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_rank;
    w_rank = ompi_comm_rank(comm);
    
    /* Determine number of elements sent per operation. */
    int seg_count = count;
    size_t typelng;
    ompi_datatype_type_size(dtype, &typelng);
    COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_future_component.future_allreduce_segsize, typelng, seg_count);
    
    OPAL_OUTPUT_VERBOSE((10, mca_coll_future_component.future_output, "In Future Allreduce seg_size %d seg_count %d count %d\n", mca_coll_future_component.future_allreduce_segsize, seg_count, count));
    int num_segments = (count + seg_count - 1) / seg_count;
    
    /* create the subcommunicators */
    mca_coll_future_module_t *future_module = (mca_coll_future_module_t *)module;
    mca_coll_future_comm_create(comm, future_module);
    ompi_communicator_t *low_comm = future_module->cached_low_comms[mca_coll_future_component.future_allreduce_low_module];
    ompi_communicator_t *up_comm = future_module->cached_up_comms[mca_coll_future_component.future_allreduce_up_module];
    int low_rank = ompi_comm_rank(low_comm);
    
    ompi_request_t *temp_request = NULL;
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = future_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    
    int root_up_rank = 0;
    int root_low_rank = 0;
    /* create sr task for the first union segment */
    mca_coll_task_t *sr = OBJ_NEW(mca_coll_task_t);
    /* setup up sr task arguments */
    int *completed = (int *)malloc(sizeof(int));
    completed[0] = 0;
    mca_allreduce_argu_t *sr_argu = malloc(sizeof(mca_allreduce_argu_t));
    mac_coll_future_set_allreduce_argu(sr_argu, sr, (char *)sbuf, (char *)rbuf, seg_count, dtype, op, root_up_rank, root_low_rank, up_comm, low_comm, num_segments, 0, w_rank, count-(num_segments-1)*seg_count, low_rank!=root_low_rank, temp_request, completed);
    /* init sr task */
    init_task(sr, mca_coll_future_allreduce_sr_task, (void *)(sr_argu));
    /* issure sr task */
    issue_task(sr);
    
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    return OMPI_SUCCESS;
}

/* sr task: issue the low level reduce of current union segment */
int mca_coll_future_allreduce_sr_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sr %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    t->low_comm->c_coll->coll_reduce((char *)t->sbuf, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
    
    /* create ir tasks for the current union segment */
    mca_coll_task_t *ir = OBJ_NEW(mca_coll_task_t);
    /* setup up ir task arguments */
    t->cur_task = ir;
    /* init ir task */
    init_task(ir, mca_coll_future_allreduce_ir_task, (void *)t);
    
    mca_coll_task_t *sr = NULL;
    /* create sr task for the next union segment if necessary */
    if (t->cur_seg+1 <= t->num_segments-1) {
        /* create sr task for the first union segment */
        sr = OBJ_NEW(mca_coll_task_t);
        /* setup up sr task arguments */
        mca_allreduce_argu_t *sr_argu = malloc(sizeof(mca_allreduce_argu_t));
        if (t->cur_seg+1 == t->num_segments-1 && t->last_seg_count != t->seg_count) {
            mac_coll_future_set_allreduce_argu(sr_argu, sr, (char *)t->sbuf+extent*t->seg_count, (char *)t->rbuf+extent*t->seg_count, t->last_seg_count, t->dtype, t->op, (t->root_up_rank+0)%ompi_comm_size(t->up_comm), t->root_low_rank, t->up_comm, t->low_comm, t->num_segments, t->cur_seg+1, t->w_rank, t->last_seg_count, t->noop, t->req, t->completed);
        }
        else {
            mac_coll_future_set_allreduce_argu(sr_argu, sr, (char *)t->sbuf+extent*t->seg_count, (char *)t->rbuf+extent*t->seg_count, t->seg_count, t->dtype, t->op, (t->root_up_rank+1)%ompi_comm_size(t->up_comm), t->root_low_rank, t->up_comm, t->low_comm, t->num_segments, t->cur_seg+1, t->w_rank, t->last_seg_count, t->noop, t->req, t->completed);
        }
        /* init sr task */
        init_task(sr, mca_coll_future_allreduce_sr_task, (void *)(sr_argu));
    }
    
    /* issue ir task */
    issue_task(ir);
    
    /* issure sr task */
    if (sr != NULL) {
        issue_task(sr);
    }
    
    return OMPI_SUCCESS;
}

static int ireduce_cb(ompi_request_t *req){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *) req->req_complete_cb_data;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ireduce_cb %d\n", t->w_rank, t->cur_seg));
    /* create ib tasks for the current union segment */
    mca_coll_task_t *ib = OBJ_NEW(mca_coll_task_t);
    /* setup up ib task arguments */
    t->cur_task = ib;
    /* init ib task */
    init_task(ib, mca_coll_future_allreduce_ib_task, (void *)t);
    issue_task(ib);
    return OMPI_SUCCESS;
}

/* ir task: issue the up level ireduce of the current union segment */
int mca_coll_future_allreduce_ir_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ir %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        /* create ib tasks for the current union segment */
        mca_coll_task_t *ib = OBJ_NEW(mca_coll_task_t);
        /* setup up ib task arguments */
        t->cur_task = ib;
        /* init ib task */
        init_task(ib, mca_coll_future_allreduce_ib_task, (void *)t);
        issue_task(ib);
        return OMPI_SUCCESS;
    }
    else {
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        ompi_request_t *ireduce_req;
        int up_rank = ompi_comm_rank(t->up_comm);
        if (up_rank == t->root_up_rank) {
            t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
        }
        else{
            t->up_comm->c_coll->coll_ireduce((char *)t->rbuf, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
        }
        ompi_request_set_callback(ireduce_req, ireduce_cb, t);
        
        return OMPI_SUCCESS;
    }
}

static int ibcast_cb(ompi_request_t *req){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *) req->req_complete_cb_data;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ibcast_cb %d\n", t->w_rank, t->cur_seg));
    /* create sb tasks for the current union segment */
    mca_coll_task_t *sb = OBJ_NEW(mca_coll_task_t);
    /* setup up sb task arguments */
    t->cur_task = sb;
    /* init sb task */
    init_task(sb, mca_coll_future_allreduce_sb_task, (void *)t);
    issue_task(sb);
    return OMPI_SUCCESS;
}


/* ib task: issue the up level ibcast of the current union segment */
int mca_coll_future_allreduce_ib_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ib %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        /* create sb tasks for the current union segment */
        mca_coll_task_t *sb = OBJ_NEW(mca_coll_task_t);
        /* setup up sb task arguments */
        t->cur_task = sb;
        /* init sb task */
        init_task(sb, mca_coll_future_allreduce_sb_task, (void *)t);
        issue_task(sb);
        return OMPI_SUCCESS;
    }
    else {
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        ompi_request_t *ibcast_req;
        t->up_comm->c_coll->coll_ibcast((char *)t->rbuf, t->seg_count, t->dtype, t->root_up_rank, t->up_comm, &ibcast_req, t->up_comm->c_coll->coll_ibcast_module);
        ompi_request_set_callback(ibcast_req, ibcast_cb, t);
        return OMPI_SUCCESS;
    }
}

/* sb task: issue the low level bcast of the current union segment */
int mca_coll_future_allreduce_sb_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sb %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    t->low_comm->c_coll->coll_bcast((char *)t->rbuf, t->seg_count, t->dtype, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
    
    int total = opal_atomic_add_fetch_32(t->completed, 1);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sb %d total %d\n", t->w_rank, t->cur_seg, total));
    if (total == t->num_segments) {
        ompi_request_t *temp_req = t->req;
        if (t->completed != NULL) {
            free(t->completed);
            t->completed = NULL;
        }
        free(t);
        ompi_request_complete(temp_req, 1);
        return OMPI_SUCCESS;
    }
    
    return OMPI_SUCCESS;
}

int
mca_coll_future_allreduce_intra_sync(const void *sbuf,
                                     void *rbuf,
                                     int count,
                                     struct ompi_datatype_t *dtype,
                                     struct ompi_op_t *op,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module){
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_rank;
    w_rank = ompi_comm_rank(comm);
    int seg_count = count;
    size_t typelng;
    ompi_datatype_type_size(dtype, &typelng);

    /* create the subcommunicators */
    mca_coll_future_module_t *future_module = (mca_coll_future_module_t *)module;
    mca_coll_future_comm_create(comm, future_module);
    ompi_communicator_t *low_comm;
    ompi_communicator_t *up_comm;
    /* auto tune is enabled */
    if (mca_coll_future_component.future_auto_tune && mca_coll_future_component.future_auto_tuned != NULL) {
        uint32_t n = future_auto_tuned_get_n(ompi_comm_size(future_module->cached_up_comms[0]));
        uint32_t c = future_auto_tuned_get_c(ompi_comm_size(future_module->cached_low_comms[0]));
        uint32_t m = future_auto_tuned_get_m(typelng * count);
        uint32_t id = n*mca_coll_future_component.future_auto_tune_c*mca_coll_future_component.future_auto_tune_m + c*mca_coll_future_component.future_auto_tune_m + m;
        uint32_t umod = mca_coll_future_component.future_auto_tuned[id].umod;
        uint32_t lmod = mca_coll_future_component.future_auto_tuned[id].lmod;
        uint32_t fs = mca_coll_future_component.future_auto_tuned[id].fs;
        uint32_t ualg = mca_coll_future_component.future_auto_tuned[id].ualg;
        uint32_t us = mca_coll_future_component.future_auto_tuned[id].us;
        /* set up umod */
        up_comm = future_module->cached_up_comms[umod];
        /* set up lmod */
        low_comm = future_module->cached_low_comms[lmod];
        /* set up fs */
        COLL_BASE_COMPUTED_SEGCOUNT((size_t)fs, typelng, seg_count);
        if (umod == 1) {
            /* set up ualg */
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_algorithm = ualg;
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_algorithm = ualg;
            /* set up us */
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_segment_size = us;
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_segment_size = us;
        }
    }
    else {
        low_comm = future_module->cached_low_comms[mca_coll_future_component.future_bcast_low_module];
        up_comm = future_module->cached_up_comms[mca_coll_future_component.future_bcast_up_module];
        COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_future_component.future_allreduce_segsize, typelng, seg_count);
    }
    
    /* Determine number of elements sent per task. */
    OPAL_OUTPUT_VERBOSE((10, mca_coll_future_component.future_output, "In Future Allreduce seg_size %d seg_count %d count %d\n", mca_coll_future_component.future_allreduce_segsize, seg_count, count));
    int num_segments = (count + seg_count - 1) / seg_count;

    int low_rank = ompi_comm_rank(low_comm);
    int root_up_rank = 0;
    int root_low_rank = 0;
    /* create t0 task for the first union segment */
    mca_coll_task_t *t0 = OBJ_NEW(mca_coll_task_t);
    /* setup up t0 task arguments */
    int *completed = (int *)malloc(sizeof(int));
    completed[0] = 0;
    mca_allreduce_argu_t *t = malloc(sizeof(mca_allreduce_argu_t));
    mac_coll_future_set_allreduce_argu(t, t0, (char *)sbuf, (char *)rbuf, seg_count, dtype, op, root_up_rank, root_low_rank, up_comm, low_comm, num_segments, 0, w_rank, count-(num_segments-1)*seg_count, low_rank!=root_low_rank, NULL, completed);
    /* init t0 task */
    init_task(t0, mca_coll_future_allreduce_t0_task, (void *)(t));
    /* issure t0 task */
    issue_task(t0);
    
    /* create t1 tasks for the current union segment */
    mca_coll_task_t *t1 = OBJ_NEW(mca_coll_task_t);
    /* setup up t1 task arguments */
    t->cur_task = t1;
    /* init t1 task */
    init_task(t1, mca_coll_future_allreduce_t1_task, (void *)t);
    /* issue t1 task */
    issue_task(t1);
    
    /* create t2 tasks for the current union segment */
    mca_coll_task_t *t2 = OBJ_NEW(mca_coll_task_t);
    /* setup up t2 task arguments */
    t->cur_task = t2;
    /* init t2 task */
    init_task(t2, mca_coll_future_allreduce_t2_task, (void *)t);
    issue_task(t2);

    /* create t3 tasks for the current union segment */
    mca_coll_task_t *t3 = OBJ_NEW(mca_coll_task_t);
    /* setup up t3 task arguments */
    t->cur_task = t3;
    /* init t3 task */
    init_task(t3, mca_coll_future_allreduce_t3_task, (void *)t);
    issue_task(t3);

    while (t->completed[0] != t->num_segments) {
        /* create t3 tasks for the current union segment */
        mca_coll_task_t *t3 = OBJ_NEW(mca_coll_task_t);
        /* setup up t3 task arguments */
        t->cur_task = t3;
        t->sbuf = (char *)t->sbuf+extent*t->seg_count;
        t->rbuf = (char *)t->rbuf+extent*t->seg_count;
        t->cur_seg = t->cur_seg + 1;
        /* init t3 task */
        init_task(t3, mca_coll_future_allreduce_t3_task, (void *)t);
        issue_task(t3);
    }
    if (t->completed != NULL) {
        free(t->completed);
        t->completed = NULL;
    }
    free(t);
    
    return OMPI_SUCCESS;
}

/* t0 task */
int mca_coll_future_allreduce_t0_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  t0 %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    t->low_comm->c_coll->coll_reduce((char *)t->sbuf, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
    return OMPI_SUCCESS;
}

/* t1 task */
int mca_coll_future_allreduce_t1_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  t1 %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *ireduce_req;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ir of cur_seg */
        if (up_rank == t->root_up_rank) {
            t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
        }
        else{
            t->up_comm->c_coll->coll_ireduce((char *)t->rbuf, (char *)t->rbuf, t->seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
        }
    }
    /* sr of cur_seg+1 */
    if (t->cur_seg <= t->num_segments-2) {
        if (t->cur_seg == t->num_segments-2 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }
        t->low_comm->c_coll->coll_reduce((char *)t->sbuf+extent*t->seg_count, (char *)t->rbuf+extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
        
    }
    if (!t->noop) {
        ompi_request_wait(&ireduce_req, MPI_STATUSES_IGNORE);
    }
    
    return OMPI_SUCCESS;
}

/* t2 task */
int mca_coll_future_allreduce_t2_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  t2 %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *reqs[2];
    int req_count = 0;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ib of cur_seg */
        t->up_comm->c_coll->coll_ibcast((char *)t->rbuf, t->seg_count, t->dtype, t->root_up_rank, t->up_comm, &(reqs[0]), t->up_comm->c_coll->coll_ibcast_module);
        req_count++;
        /* ir of cur_seg+1 */
        if (t->cur_seg <= t->num_segments-2) {
            if (t->cur_seg == t->num_segments-2 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            if (up_rank == t->root_up_rank) {
                t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *)t->rbuf+extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &(reqs[1]), t->up_comm->c_coll->coll_ireduce_module);
            }
            else{
                t->up_comm->c_coll->coll_ireduce((char *)t->rbuf+extent*t->seg_count, (char *)t->rbuf+extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &(reqs[1]), t->up_comm->c_coll->coll_ireduce_module);
            }
            req_count++;
        }
    }
    /* sr of cur_seg+2 */
    if (t->cur_seg <= t->num_segments-3) {
        if (t->cur_seg == t->num_segments-3 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }
        t->low_comm->c_coll->coll_reduce((char *)t->sbuf+2*extent*t->seg_count, (char *)t->rbuf+2*extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
    }
    if (!t->noop && req_count > 0) {
        ompi_request_wait_all(req_count, reqs, MPI_STATUSES_IGNORE);
    }
    
    
    return OMPI_SUCCESS;
}

/* t3 task */
int mca_coll_future_allreduce_t3_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  t3 %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *reqs[2];
    int req_count = 0;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ib of cur_seg+1 */
        if (t->cur_seg <= t->num_segments-2) {
            if (t->cur_seg == t->num_segments-2 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            t->up_comm->c_coll->coll_ibcast((char *)t->rbuf+extent*t->seg_count, t->seg_count, t->dtype, t->root_up_rank, t->up_comm, &(reqs[0]), t->up_comm->c_coll->coll_ibcast_module);
            req_count++;
        }
        /* ir of cur_seg+2 */
        if (t->cur_seg <= t->num_segments-3) {
            if (t->cur_seg == t->num_segments-3 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            if (up_rank == t->root_up_rank) {
                t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *)t->rbuf+2*extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &(reqs[1]), t->up_comm->c_coll->coll_ireduce_module);
            }
            else{
                t->up_comm->c_coll->coll_ireduce((char *)t->rbuf+2*extent*t->seg_count, (char *)t->rbuf+2*extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &(reqs[1]), t->up_comm->c_coll->coll_ireduce_module);
            }
            req_count++;
        }
    }
    /* sr of cur_seg+3 */
    if (t->cur_seg <= t->num_segments-4) {
        if (t->cur_seg == t->num_segments-4 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }
        t->low_comm->c_coll->coll_reduce((char *)t->sbuf+3*extent*t->seg_count, (char *)t->rbuf+3*extent*t->seg_count, tmp_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
    }
    /* sb of cur_seg */
    t->low_comm->c_coll->coll_bcast((char *)t->rbuf, t->seg_count, t->dtype, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
    if (!t->noop && req_count > 0) {
        ompi_request_wait_all(req_count, reqs, MPI_STATUSES_IGNORE);
    }
    
    t->completed[0]++;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  t3 %d total %d\n", t->w_rank, t->cur_seg, t->completed[0]));

    return OMPI_SUCCESS;
}

