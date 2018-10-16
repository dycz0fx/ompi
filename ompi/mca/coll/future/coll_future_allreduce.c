#include "coll_future.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_future_trigger.h"

void mac_coll_future_set_argu(mca_allreduce_argu_t *argu,
                              mca_coll_task_t *cur_task,
                              void *sbuf,
                              void *rbuf,
                              int up_seg_count,
                              int low_seg_count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              int root_up_rank,
                              int root_low_rank,
                              struct ompi_communicator_t *up_comm,
                              struct ompi_communicator_t *low_comm,
                              int up_num,
                              int low_num,
                              int num_segments,
                              int cur_seg,
                              int w_rank,
                              int last_seg_count,
                              bool noop,
                              ompi_request_t *req,
                              int *completed,
                              int *ongoing){
    argu->cur_task = cur_task;
    argu->sbuf = sbuf;
    argu->rbuf = rbuf;
    argu->up_seg_count = up_seg_count;
    argu->low_seg_count = low_seg_count;
    argu->dtype = dtype;
    argu->op = op;
    argu->root_up_rank = root_up_rank;
    argu->root_low_rank = root_low_rank;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->up_num = up_num;
    argu->low_num = low_num;
    argu->num_segments = num_segments;
    argu->cur_seg = cur_seg;
    argu->w_rank = w_rank;
    argu->last_seg_count = last_seg_count;
    argu->noop = noop;
    argu->req = req;
    argu->completed = completed;
    argu->ongoing = ongoing;
}

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
    int w_size, w_rank;
    w_size = ompi_comm_size(comm);
    w_rank = ompi_comm_rank(comm);
    int up_seg_count = mca_coll_future_component.future_allreduce_up_count;
    int low_seg_count = mca_coll_future_component.future_allreduce_low_count;
    mca_coll_future_reset_seg_count(&up_seg_count, &low_seg_count, &count);
    OPAL_OUTPUT_VERBOSE((10, mca_coll_future_component.future_output, "In Future Allreduce up_seg_count %d low_seg_count %d count %d\n", up_seg_count, low_seg_count, count));
    int max_seg_count = (up_seg_count > low_seg_count) ? up_seg_count : low_seg_count;
    int up_num = max_seg_count / up_seg_count;
    int low_num = max_seg_count / low_seg_count;
    int num_segments = (count + max_seg_count - 1) / max_seg_count;

    //TODO: wrap this into a function (communicator creation)
    ompi_communicator_t *sm_comm;
    ompi_communicator_t *leader_comm;
    int *vranks;
    int sm_rank, sm_size;
    int leader_rank;
    mca_coll_future_module_t *future_module = (mca_coll_future_module_t *)module;
    /* use cached communicators if possible */
    if (future_module->cached_comm == comm && future_module->cached_sm_comm != NULL && future_module->cached_leader_comm != NULL && future_module->cached_vranks != NULL) {
        sm_comm = future_module->cached_sm_comm;
        leader_comm = future_module->cached_leader_comm;
        vranks = future_module->cached_vranks;
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);
        leader_rank = ompi_comm_rank(leader_comm);
    }
    /* create communicators if there is no cached communicator */
    else {
        /* create sm_comm which contain all the process on a node */
        const int *origin_priority = NULL;
        //const int *tmp = NULL;
        /* lower future module priority */
        int future_var_id;
        int tmp_future_priority = 0;
        int tmp_future_origin = 0;
        mca_base_var_find_by_name("coll_future_priority", &future_var_id);
        mca_base_var_get_value(future_var_id, &origin_priority, NULL, NULL);
        tmp_future_origin = *origin_priority;
        mca_base_var_set_flag(future_var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(future_var_id, &tmp_future_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        comm->c_coll->coll_allreduce = ompi_coll_base_allreduce_intra_recursivedoubling;
        
        int var_id;
        int tmp_priority = 60;
        int tmp_origin = 0;
        mca_base_var_find_by_name("coll_shared_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] shared_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("sm_priority after set %d %d\n", *tmp);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &sm_comm);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] sm_priority set back %d\n", w_rank, *tmp);
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);
        
        /* create leader_comm which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_adapt_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] adapt_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("adapt_priority after set %d %d\n", *tmp);
        ompi_comm_split(comm, sm_rank, w_rank, &leader_comm, false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] adapt_priority set back %d\n", w_rank, *tmp);
        leader_rank = ompi_comm_rank(leader_comm);
        
        vranks = malloc(sizeof(int) * w_size);
        /* do allgather to gather vrank from each process so every process will know other processes vrank*/
        int vrank = sm_size * leader_rank + sm_rank;
        comm->c_coll->coll_allgather(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm, comm->c_coll->coll_allgather_module);
        future_module->cached_comm = comm;
        future_module->cached_sm_comm = sm_comm;
        future_module->cached_leader_comm = leader_comm;
        future_module->cached_vranks = vranks;
        
        mca_base_var_set_value(future_var_id, &tmp_future_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        comm->c_coll->coll_allreduce = mca_coll_future_allreduce_intra;
    }
    
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
    int *ongoing = (int *)malloc(sizeof(int));;
    mca_allreduce_argu_t *sr_argu = malloc(sizeof(mca_allreduce_argu_t));
    mac_coll_future_set_argu(sr_argu, sr, (char *)sbuf, (char *)rbuf, up_seg_count, low_seg_count, dtype, op, root_up_rank, root_low_rank, leader_comm, sm_comm, up_num, low_num, num_segments, 0, w_rank, count-(num_segments-1)*max_seg_count, sm_rank!=root_low_rank, temp_request, completed, ongoing);
    /* init sr task */
    init_task(sr, mca_coll_future_sr_task, (void *)(sr_argu));
    /* issure sr task */
    issue_task(sr);
    
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    return OMPI_SUCCESS;
}

/* sr task: issue the low level reduce of current union segment */
int mca_coll_future_sr_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sr %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    int i;
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    for (i=0; i<t->low_num; i++) {
        t->low_comm->c_coll->coll_reduce((char *)t->sbuf+extent*t->low_seg_count*i, (char *)t->rbuf+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->op, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_reduce_module);
    }

    /* create ir tasks for the current union segment */
    mca_coll_task_t *ir = OBJ_NEW(mca_coll_task_t);
    /* setup up ir task arguments */
    t->cur_task = ir;
    /* init ir task */
    init_task(ir, mca_coll_future_ir_task, (void *)t);
    
    mca_coll_task_t *sr = NULL;
    /* create sr task for the next union segment if necessary */
    if (t->cur_seg+1 <= t->num_segments-1) {
        /* create sr task for the first union segment */
        sr = OBJ_NEW(mca_coll_task_t);
        /* setup up sr task arguments */
        int *ongoing = (int *)malloc(sizeof(int));;
        mca_allreduce_argu_t *sr_argu = malloc(sizeof(mca_allreduce_argu_t));
        int max_seg_count = (t->up_seg_count > t->low_seg_count) ? t->up_seg_count : t->low_seg_count;
        if (t->cur_seg+1 == t->num_segments-1 && t->last_seg_count != max_seg_count) {
            mac_coll_future_set_argu(sr_argu, sr, (char *)t->sbuf+extent*max_seg_count, (char *)t->rbuf+extent*max_seg_count, t->last_seg_count, t->last_seg_count, t->dtype, t->op, (t->root_up_rank+1)%ompi_comm_size(t->up_comm), t->root_low_rank, t->up_comm, t->low_comm, 1, 1, t->num_segments, t->cur_seg+1, t->w_rank, t->last_seg_count, t->noop, t->req, t->completed, ongoing);
        }
        else {
            mac_coll_future_set_argu(sr_argu, sr, (char *)t->sbuf+extent*max_seg_count, (char *)t->rbuf+extent*max_seg_count, t->up_seg_count, t->low_seg_count, t->dtype, t->op, (t->root_up_rank+1)%ompi_comm_size(t->up_comm), t->root_low_rank, t->up_comm, t->low_comm, t->up_num, t->low_num, t->num_segments, t->cur_seg+1, t->w_rank, t->last_seg_count, t->noop, t->req, t->completed, ongoing);
        }
        /* init sr task */
        init_task(sr, mca_coll_future_sr_task, (void *)(sr_argu));
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
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ireduce_cb %d ongoing %d \n", t->w_rank, t->cur_seg, *(t->ongoing)));
    int remain = opal_atomic_sub_fetch_32(t->ongoing, 1);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ireduce_cb %d remain %d\n", t->w_rank, t->cur_seg, remain));
    if (remain == 0) {
        /* create ib tasks for the current union segment */
        mca_coll_task_t *ib = OBJ_NEW(mca_coll_task_t);
        /* setup up ib task arguments */
        t->cur_task = ib;
        /* init ib task */
        init_task(ib, mca_coll_future_ib_task, (void *)t);
        issue_task(ib);
    }
    return OMPI_SUCCESS;
}

/* ir task: issue the up level ireduce of the current union segment */
int mca_coll_future_ir_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ir %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        /* create ib tasks for the current union segment */
        mca_coll_task_t *ib = OBJ_NEW(mca_coll_task_t);
        /* setup up ib task arguments */
        t->cur_task = ib;
        /* init ib task */
        init_task(ib, mca_coll_future_ib_task, (void *)t);
        issue_task(ib);
        return OMPI_SUCCESS;
    }
    else {
        *(t->ongoing) = t->up_num;
        int i;
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        for (i=0; i<t->up_num; i++) {
            ompi_request_t *ireduce_req;
            int up_rank = ompi_comm_rank(t->up_comm);
            if (up_rank == t->root_up_rank) {
                t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *)t->rbuf+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
            }
            else{
                t->up_comm->c_coll->coll_ireduce((char *)t->rbuf+extent*t->up_seg_count*i, (char *)t->rbuf+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->op, t->root_up_rank, t->up_comm, &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
            }
            ompi_request_set_callback(ireduce_req, ireduce_cb, t);
        }
        return OMPI_SUCCESS;
    }
}

static int ibcast_cb(ompi_request_t *req){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *) req->req_complete_cb_data;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ibcast_cb %d ongoing %d \n", t->w_rank, t->cur_seg, *(t->ongoing)));
    int remain = opal_atomic_sub_fetch_32(t->ongoing, 1);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ibcast_cb %d remain %d\n", t->w_rank, t->cur_seg, remain));
    if (remain == 0) {
        /* create sb tasks for the current union segment */
        mca_coll_task_t *sb = OBJ_NEW(mca_coll_task_t);
        /* setup up sb task arguments */
        t->cur_task = sb;
        /* init sb task */
        init_task(sb, mca_coll_future_sb_task, (void *)t);
        issue_task(sb);
    }
    return OMPI_SUCCESS;
}


/* ib task: issue the up level ibcast of the current union segment */
int mca_coll_future_ib_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  ib %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        /* create sb tasks for the current union segment */
        mca_coll_task_t *sb = OBJ_NEW(mca_coll_task_t);
        /* setup up sb task arguments */
        t->cur_task = sb;
        /* init sb task */
        init_task(sb, mca_coll_future_sb_task, (void *)t);
        issue_task(sb);
        return OMPI_SUCCESS;
    }
    else {
        *(t->ongoing) = t->up_num;
        int i;
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        for (i=0; i<t->up_num; i++) {
            ompi_request_t *ibcast_req;
            t->up_comm->c_coll->coll_ibcast((char *)t->rbuf+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->root_up_rank, t->up_comm, &ibcast_req, t->up_comm->c_coll->coll_ibcast_module);
            ompi_request_set_callback(ibcast_req, ibcast_cb, t);
        }
        return OMPI_SUCCESS;
    }
}

/* sb task: issue the low level bcast of the current union segment */
int mca_coll_future_sb_task(void *task_argu){
    mca_allreduce_argu_t *t = (mca_allreduce_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sb %d r_buf %d\n", t->w_rank, t->cur_seg, ((int *)t->rbuf)[1]));
    OBJ_RELEASE(t->cur_task);
    int i;
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    for (i=0; i<t->low_num; i++) {
        t->low_comm->c_coll->coll_bcast((char *)t->rbuf+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
    }
    
    int total = opal_atomic_add_fetch_32(t->completed, 1);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] Future Allreduce:  sb %d total %d\n", t->w_rank, t->cur_seg, total));
    if (total == t->num_segments) {
        ompi_request_t *temp_req = t->req;
        if (t->completed != NULL) {
            free(t->completed);
            t->completed = NULL;
        }
        if (t->ongoing != NULL) {
            free(t->ongoing);
            t->ongoing = NULL;
        }
        free(t);
        ompi_request_complete(temp_req, 1);
        return OMPI_SUCCESS;
    }

    return OMPI_SUCCESS;
}
