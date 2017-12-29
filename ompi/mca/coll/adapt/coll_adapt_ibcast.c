/*
 * Copyright (c) 2014-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ompi/mca/pml/pml.h"
#include "coll_adapt.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "opal/util/bit_ops.h"
#include "opal/sys/atomic.h"                //atomic
#include "ompi/mca/pml/ob1/pml_ob1.h"       //dump

#if OPAL_CUDA_SUPPORT
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "opal/mca/common/cuda/common_cuda.h"
#endif

/* Bcast algorithm variables */
static int coll_adapt_ibcast_algorithm = 0;
static size_t coll_adapt_ibcast_segment_size = ADAPT_DEFAULT_SEGSIZE;
static int coll_adapt_ibcast_max_send_requests = 2;
static int coll_adapt_ibcast_max_recv_requests = 3;
static opal_free_list_t *coll_adapt_ibcast_context_free_list = NULL;
static int32_t coll_adapt_ibcast_context_free_list_enabled = 0;


typedef int (*mca_coll_adapt_ibcast_fn_t)(
void *buff,
int count,
struct ompi_datatype_t *datatype,
int root,
struct ompi_communicator_t *comm,
ompi_request_t ** request,
mca_coll_base_module_t *module,
int ibcast_tag
);

static mca_coll_adapt_algorithm_index_t mca_coll_adapt_ibcast_algorithm_index[] = {
    {0, (uintptr_t)mca_coll_adapt_ibcast_tuned},
    {1, (uintptr_t)mca_coll_adapt_ibcast_binomial},
    {2, (uintptr_t)mca_coll_adapt_ibcast_in_order_binomial},
    {3, (uintptr_t)mca_coll_adapt_ibcast_binary},
    {4, (uintptr_t)mca_coll_adapt_ibcast_pipeline},
    {5, (uintptr_t)mca_coll_adapt_ibcast_chain},
    {6, (uintptr_t)mca_coll_adapt_ibcast_linear},
    {7, (uintptr_t)mca_coll_adapt_ibcast_topoaware_linear},
    {8, (uintptr_t)mca_coll_adapt_ibcast_topoaware_chain},
};

#if OPAL_CUDA_SUPPORT
static int bcast_init_cpu_buff(mca_coll_adapt_constant_bcast_context_t *con);
static int bcast_send_context_async_memcpy_callback(mca_coll_adapt_bcast_context_t *send_context);
static int update_ref_count(mca_coll_adapt_bcast_context_t *context);
#endif

int mca_coll_adapt_ibcast_init(void)
{
    mca_base_component_t *c = &mca_coll_adapt_component.super.collm_version;
    
    mca_base_component_var_register(c, "bcast_algorithm",
                                    "Algorithm of broadcast",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ibcast_algorithm);
    
    mca_base_component_var_register(c, "bcast_segment_size",
                                    "Segment size in bytes used by default for bcast algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ibcast_segment_size);
    
    mca_base_component_var_register(c, "bcast_max_send_requests",
                                    "Maximum number of send requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ibcast_max_send_requests);
    
    mca_base_component_var_register(c, "bcast_max_recv_requests",
                                    "Maximum number of receive requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ibcast_max_recv_requests);
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_fini(void)
{
    if (NULL != coll_adapt_ibcast_context_free_list) {
        OBJ_RELEASE(coll_adapt_ibcast_context_free_list);
        coll_adapt_ibcast_context_free_list = NULL;
        coll_adapt_ibcast_context_free_list_enabled = 0;
        OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "bcast fini\n"));
    }
    return OMPI_SUCCESS;
}

static int ibcast_request_fini(mca_coll_adapt_bcast_context_t *context)
{
    ompi_request_t *temp_req = context->con->request;
    if (context->con->tree->tree_nextsize != 0) {
        free(context->con->send_array);
    }
    if (context->con->num_segs !=0) {
        free(context->con->recv_array);
    }
#if OPAL_CUDA_SUPPORT
    int free_count = 0;
    int k;
    mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_cpu_mpool;
    mca_common_cuda_sync_memcpy_stream();
    if (con->cpu_buff_list != NULL) {
        for (k = 0; k < con->num_segs; k++) {
            if (con->cpu_buff_list[k] != NULL) {
                mpool->mpool_free(mpool, con->cpu_buff_list[k]);
                free_count ++;
            }
        }
        free(con->cpu_buff_list);
        con->cpu_buff_list = NULL;
    }
    if (con->cpu_buff_memcpy_flags != NULL) {
        free(con->cpu_buff_memcpy_flags);
    }
    if (con->cpu_buff_list_ref_count != NULL) {
        free(con->cpu_buff_list_ref_count);
    }
#endif
    ompi_coll_base_topo_destroy_tree(&(context->con->tree));
    OBJ_RELEASE(context->con->mutex);
    OBJ_RELEASE(context->con);
    OBJ_RELEASE(context->con);
    opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
    ompi_request_complete(temp_req, 1);
    
    return OMPI_SUCCESS;
}

/* send call back */
static int send_cb(ompi_request_t *req)
{
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
#if OPAL_CUDA_SUPPORT
    mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_cpu_mpool;
    ompi_coll_tree_t *tree = context->con->tree;
#endif
    
    int err;
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p root %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->con->root));
    
    OPAL_THREAD_LOCK(context->con->mutex);
#if OPAL_CUDA_SUPPORT
    /*  check if cpu_buff_list can be released */
    update_ref_count(context);
#endif
    int sent_id = context->con->send_array[context->child_id];
    /* has fragments in recv_array can be sent */
    if (sent_id < context->con->num_recv_segs) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_array[sent_id];
        mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
        send_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        send_context->frag_id = new_id;
        send_context->child_id = context->child_id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        int send_count = send_context->con->seg_count;
        if (new_id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - new_id * send_context->con->seg_count;
        }
        ++(send_context->con->send_array[send_context->child_id]);
        char *send_buff = send_context->buff;
#if OPAL_CUDA_SUPPORT
        if (send_context->con->gpu_use_cpu_buff) {
            assert (tree->topo_flags == 0 && tree->tree_prev_topo_flags == -1);
            if (tree->tree_next_topo_flags[context->child_id] != 2) {
                /* send to socket or node leader, use cpu buff */
                assert(context->con->cpu_buff_list != NULL);
                if (context->con->cpu_buff_memcpy_flags[new_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                    context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
                    ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->con->cpu_buff_list[new_id], (char*)send_context->buff);
                    context->con->cpu_buff_memcpy_flags[new_id] = CPU_BUFFER_MEMCPY_DONE;
                }
                send_buff = context->con->cpu_buff_list[new_id];
                if (context->con->cpu_buff_memcpy_flags[new_id] != CPU_BUFFER_MEMCPY_DONE) {
                    send_context->send_count = send_count;
                    send_context->buff = send_buff;
                    send_context->flags = COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST;
                    send_context->cuda_callback = bcast_send_context_async_memcpy_callback;
                    send_context->debug_flag = 999;
                    mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)send_context);
                    goto SEND_CB_SKIP_SEND;
                }
            }
        }
#endif
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d tag %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (send_context->con->ibcast_tag << 16) + new_id));
        err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ibcast_tag << 16) + new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        
        /* invoke send call back */
        OPAL_THREAD_UNLOCK(context->con->mutex);
        ompi_request_set_callback(send_req, send_cb, send_context);
        OPAL_THREAD_LOCK(context->con->mutex);
    }
    
SEND_CB_SKIP_SEND: ;
    int num_sent = ++(context->con->num_sent_segs);
    int num_recv_fini_t = context->con->num_recv_fini;
    int rank = ompi_comm_rank(context->con->comm);
    opal_mutex_t * mutex_temp = context->con->mutex;
    /* check whether signal the condition */
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Singal in send\n", ompi_comm_rank(context->con->comm)));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else {
        OBJ_RELEASE(context->con);
        opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

/* receive call back */
static int recv_cb(ompi_request_t *req){
    /* get necessary info from request */
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
    int err, i;
    
#if OPAL_CUDA_SUPPORT
    mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_cpu_mpool;
    ompi_coll_tree_t *tree = context->con->tree;
#endif
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p root %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->con->root));
    
    /* store the frag_id to seg array */
    OPAL_THREAD_LOCK(context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    context->con->recv_array[num_recv_segs_t-1] = context->frag_id;
    
    int new_id = num_recv_segs_t + coll_adapt_ibcast_max_recv_requests - 1;
    /* receive new segment */
    if (new_id < context->con->num_segs) {
        ompi_request_t *recv_req;
        /* get new context item from free list */
        mca_coll_adapt_bcast_context_t * recv_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
        recv_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        char *recv_buff = recv_context->buff;
#if OPAL_CUDA_SUPPORT
        if (recv_context->con->gpu_use_cpu_buff) {
            if (tree->topo_flags == 1 || tree->topo_flags == 0) {
                /* node / socket leader, receive to cpu mem */
                context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
                recv_buff = context->con->cpu_buff_list[new_id];
            }
        }
#endif
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d tag %d\n", ompi_comm_rank(context->con->comm), context->frag_id, context->peer, (void *)recv_buff, recv_count, (recv_context->con->ibcast_tag << 16) + recv_context->frag_id));
        MCA_PML_CALL(irecv(recv_buff, recv_count, recv_context->con->datatype, recv_context->peer, (recv_context->con->ibcast_tag << 16) + recv_context->frag_id, recv_context->con->comm, &recv_req));
        
        /* invoke recvive call back */
        OPAL_THREAD_UNLOCK(context->con->mutex);
        ompi_request_set_callback(recv_req, recv_cb, recv_context);
        OPAL_THREAD_LOCK(context->con->mutex);
    }
    
#if OPAL_CUDA_SUPPORT
    if (context->con->gpu_use_cpu_buff && (tree->topo_flags == 1 || tree->topo_flags == 0) ) {
        /* node/socket leader move received data from CPU buffer back to GPU memory */
        assert(context->con->cpu_buff_list[context->frag_id] != NULL);
        if (context->con->cpu_buff_memcpy_flags[context->frag_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
            int copy_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                copy_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            context->con->datatype->super.flags |= OPAL_DATATYPE_FLAG_GPU_MEMCPY_ASYNC;
            ompi_datatype_copy_content_same_ddt(context->con->datatype, copy_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
            context->con->datatype->super.flags &= ~OPAL_DATATYPE_FLAG_GPU_MEMCPY_ASYNC;
            context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_PENDING;
        }
    }
#endif
    
    /* send segment to its children */
    for (i = 0; i < context->con->tree->tree_nextsize; i++) {
        /* if can send the segment now means the only segment need to be sent is the just arrived one */
        if (num_recv_segs_t-1 == context->con->send_array[i]) {
            ompi_request_t *send_req;
            int send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            
            mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->tree->tree_next[i];
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_array[i]);
            char *send_buff = send_context->buff;
#if OPAL_CUDA_SUPPORT
            if (send_context->con->gpu_use_cpu_buff) {
                if (tree->topo_flags == 1) {
                    /* socket leader, move data from cpu buff into send_buff */
                    send_context->send_count = send_count;
                    send_context->buff = context->buff;
                    send_context->flags = COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST;
                    send_context->cuda_callback = bcast_send_context_async_memcpy_callback;
                    mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)send_context);
                    continue;
                    /* do not send now, send in call back */
                    
                } else if (tree->topo_flags == 0) {  /* node leader */
                    assert(context->con->cpu_buff_list[context->frag_id] != NULL);
                    if (tree->tree_next_topo_flags[i] == 2) {
                        /* send to process in his group */
                        send_context->send_count = send_count;
                        send_context->buff = context->buff;
                        send_context->flags = COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST;
                        send_context->cuda_callback = bcast_send_context_async_memcpy_callback;
                        mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)send_context);
                        continue;
                        /* do not send now, send in call back */
                        
                    } else {
                        /* send to node/socket leader, use cpu buff */
                        send_buff = context->con->cpu_buff_list[context->frag_id];
                    }
                }
            }
#endif
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d tag %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (send_context->con->ibcast_tag << 16) + send_context->frag_id));
            err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ibcast_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            /* invoke send call back */
            OPAL_THREAD_UNLOCK(context->con->mutex);
            ompi_request_set_callback(send_req, send_cb, send_context);
            OPAL_THREAD_LOCK(context->con->mutex);
        }
    }
    
    int num_sent = context->con->num_sent_segs;
    int num_recv_fini_t = ++(context->con->num_recv_fini);
    int rank = ompi_comm_rank(context->con->comm);
    opal_mutex_t * mutex_temp = context->con->mutex;
    
    /* if this is leaf and has received all the segments */
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm)));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else{
        OBJ_RELEASE(context->con);
        opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    if (count == 0) {
        ompi_request_t *temp_request;
        temp_request = OBJ_NEW(ompi_request_t);
        OMPI_REQUEST_INIT(temp_request, false);
        temp_request->req_type = 0;
        temp_request->req_free = adapt_request_free;
        temp_request->req_status.MPI_SOURCE = 0;
        temp_request->req_status.MPI_TAG = 0;
        temp_request->req_status.MPI_ERROR = 0;
        temp_request->req_status._cancelled = 0;
        temp_request->req_status._ucount = 0;
        ompi_request_complete(temp_request, 1);
        *request = temp_request;
        return MPI_SUCCESS;
    }
    else {
        int rank = ompi_comm_rank(comm);
        if (rank == root) {
            OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "ibcast root %d, algorithm %d, coll_adapt_ibcast_segment_size %zu, coll_adapt_ibcast_max_send_requests %d, coll_adapt_ibcast_max_recv_requests %d\n", root, coll_adapt_ibcast_algorithm, coll_adapt_ibcast_segment_size, coll_adapt_ibcast_max_send_requests, coll_adapt_ibcast_max_recv_requests));
        }
        int ibcast_tag = opal_atomic_add_32(&(comm->c_ibcast_tag), 1);
        ibcast_tag = ibcast_tag % 4096;
#if OPAL_CUDA_SUPPORT
        if (1 == mca_common_is_cuda_buffer(buff)) {
            return mca_coll_adapt_ibcast_cuda(buff, count, datatype, root, comm, request, module, ibcast_tag);
        }
#endif
        mca_coll_adapt_ibcast_fn_t bcast_func = (mca_coll_adapt_ibcast_fn_t)mca_coll_adapt_ibcast_algorithm_index[coll_adapt_ibcast_algorithm].algorithm_fn_ptr;
        return bcast_func(buff, count, datatype, root, comm, request, module, ibcast_tag);
    }
}

#if OPAL_CUDA_SUPPORT
int mca_coll_adapt_ibcast_cuda(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag) {
    coll_adapt_ibcast_segment_size = 524288;
    if (0 == mca_coll_adapt_component.coll_adapt_cuda_enabled) {
        coll_adapt_cuda_init();
    }
    if (0 == mca_common_cuda_is_stage_three_init()) {
        return mca_coll_adapt_ibcast_pipeline(buff, count, datatype, root, comm, request, module, ibcast_tag);
    } else {
        mca_coll_base_comm_t *coll_comm = module->base_data;
        if( !( (coll_comm->cached_topochain) && (coll_comm->cached_topochain_root == root) ) ) {
            if( coll_comm->cached_topochain ) {
                /* destroy previous binomial if defined */
                ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topochain) );
            }
            ompi_coll_topo_gpu_t *gpu_topo = (ompi_coll_topo_gpu_t *)malloc(sizeof(ompi_coll_topo_gpu_t));
            coll_adapt_cuda_get_gpu_topo(gpu_topo);
            coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 4, 1, (void*)gpu_topo);
            coll_comm->cached_topochain_root = root;
            coll_adapt_cuda_free_gpu_topo(gpu_topo);
            free(gpu_topo);
        }
        else {
        }
        return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_topochain, coll_adapt_ibcast_segment_size, ibcast_tag, 1);
    }
}
#endif

int mca_coll_adapt_ibcast_tuned(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    /* Just Place holder for future auto tuned ibcast */
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "tuned\n"));
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
    
}

/* Binomial tree with pipeline */
int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "tuned\n"));
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}


int mca_coll_adapt_ibcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_tree(2, comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_chain(1, comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_chain(4, comm, root);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    int fanout = ompi_comm_size(comm) - 1;
    ompi_coll_tree_t* tree;
    if (fanout > 1) {
        tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    }
    else{
        tree = ompi_coll_base_topo_build_chain(1, comm, root);
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_topoaware_linear(comm, root, module, 3, 0, NULL);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    size_t seg_size = ADAPT_DEFAULT_SEGSIZE;
    if (coll_adapt_ibcast_segment_size > 0) {
        seg_size = coll_adapt_ibcast_segment_size;
    }
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_topoaware_chain(comm, root, module, 3, 0, NULL);
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, seg_size, ibcast_tag, 0);
}

int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t seg_size, int ibcast_tag, int gpu_use_cpu_buff){
    int i, j;       /* temp variable for iteration */
    int size;       /* size of the communicator */
    int rank;       /* rank of this node */
    int err;        /* record return value */
    int min;        /* the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM */
    
    int seg_count = count;      /* number of datatype in a segment */
    size_t type_size;           /* the size of a datatype */
    size_t real_seg_size;       /* the real size of a segment */
    ptrdiff_t extent, lb;
    int num_segs;               /* the number of segments */
                                 
    ompi_request_t * temp_request = NULL;  /* the request be passed outside */
    opal_mutex_t * mutex;
    int *recv_array = NULL;   /* store those segments which are received */
    int *send_array = NULL;   /* record how many isend has been issued for every child */
    
    /* set up free list */
    if (0 == coll_adapt_ibcast_context_free_list_enabled) {
        int32_t context_free_list_enabled = opal_atomic_add_32(&(coll_adapt_ibcast_context_free_list_enabled), 1);
        if (1 == context_free_list_enabled) {
            coll_adapt_ibcast_context_free_list = OBJ_NEW(opal_free_list_t);
            opal_free_list_init(coll_adapt_ibcast_context_free_list,
                                sizeof(mca_coll_adapt_bcast_context_t),
                                opal_cache_line_size,
                                OBJ_CLASS(mca_coll_adapt_bcast_context_t),
                                0,opal_cache_line_size,
                                mca_coll_adapt_component.adapt_context_free_list_min,
                                mca_coll_adapt_component.adapt_context_free_list_max,
                                mca_coll_adapt_component.adapt_context_free_list_inc,
                                NULL, 0, NULL, NULL, NULL);
        }
    }
    
    /* set up cpu mpool */
#if OPAL_CUDA_SUPPORT
    mca_mpool_base_module_t *mpool = NULL;
    if (gpu_use_cpu_buff) {
        if (mca_coll_adapt_component.pined_cpu_mpool == NULL) {
            mca_coll_adapt_component.pined_cpu_mpool = coll_adapt_cuda_mpool_create(MPOOL_CPU);
        }
        mpool = mca_coll_adapt_component.pined_cpu_mpool;
        assert(mpool != NULL);
    }
#endif
    
    /* set up request */
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    *request = temp_request;
    
    /* set up mutex */
    mutex = OBJ_NEW(opal_mutex_t);
    
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    
    /* determine number of elements sent per operation */
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    
    ompi_datatype_get_extent(datatype, &lb, &extent);
    num_segs = (count + seg_count - 1) / seg_count;
    real_seg_size = (ptrdiff_t)seg_count * extent;
    
    /* set memory for recv_array and send_array, created on heap becasue they are needed to be accessed by other functions, callback function */
    if (num_segs!=0) {
        recv_array = (int *)malloc(sizeof(int) * num_segs);
    }
    if (tree->tree_nextsize!=0) {
        send_array = (int *)malloc(sizeof(int) * tree->tree_nextsize);
    }
    
    /* set constant context for send and recv call back */
    mca_coll_adapt_constant_bcast_context_t *con = OBJ_NEW(mca_coll_adapt_constant_bcast_context_t);
    con->root = root;
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = datatype;
    con->comm = comm;
    con->real_seg_size = real_seg_size;
    con->num_segs = num_segs;
    con->recv_array = recv_array;
    con->num_recv_segs = 0;
    con->num_recv_fini = 0;
    con->send_array = send_array;
    con->num_sent_segs = 0;
    con->mutex = mutex;
    con->request = temp_request;
    con->tree = tree;
    con->ibcast_tag = ibcast_tag;
    
#if OPAL_CUDA_SUPPORT
    con->cpu_buff_list = NULL;
    con->cpu_buff_memcpy_flags = NULL;
    con->cpu_buff_list_ref_count = NULL;
    con->gpu_use_cpu_buff = gpu_use_cpu_buff;
#endif
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"[%d, %" PRIx64 "]: Ibcast, root %d, tag %d\n", rank, gettid(), root, ibcast_tag));
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"[%d, %" PRIx64 "]: con->mutex = %p, num_children = %d, num_segs = %d, real_seg_size = %d, seg_count = %d, tree_adreess = %p\n", rank, gettid(), (void *)con->mutex, tree->tree_nextsize, num_segs, (int)real_seg_size, seg_count, (void *)con->tree));
    
    OPAL_THREAD_LOCK(mutex);
    
    /* if root, send segment to every children. */
    if (rank == root){
        
#if OPAL_CUDA_SUPPORT
        tree->tree_prev_topo_flags = -1;
#endif
        /* handle the situation when num_segs < SEND_NUM */
        if (num_segs <= coll_adapt_ibcast_max_send_requests) {
            min = num_segs;
        }
        else{
            min = coll_adapt_ibcast_max_send_requests;
        }
        
        /* set recv_array, root has already had all the segments */
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = i;
        }
        con->num_recv_segs = num_segs;
        /* set send_array, has not sent any segments */
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = coll_adapt_ibcast_max_send_requests;
        }
        
        ompi_request_t *send_req;
        int send_count = seg_count;             /* number of datatype in each send */
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                send_count = count - i * seg_count;
            }
            for (j=0; j<tree->tree_nextsize; j++) {
                mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
                context->buff = (char *)buff + i * real_seg_size;
                context->frag_id = i;
                context->child_id = j;                  /* the id of peer in in children_list */
                context->peer = tree->tree_next[j];     /* the actural rank of the peer */
                context->con = con;
                OBJ_RETAIN(con);
                
                char* send_buff = context->buff;
#if OPAL_CUDA_SUPPORT
                if (con->gpu_use_cpu_buff) {
                    if (tree->topo_flags != -1 && tree->tree_next_topo_flags[j] != 2) {
                        /* send to socket or node leader, send through cpu mem */
                        if (con->cpu_buff_list == NULL) {
                            bcast_init_cpu_buff(con);
                        }
                        if (con->cpu_buff_memcpy_flags[i] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                            con->cpu_buff_list[i] = mpool->mpool_alloc(mpool, sizeof(char)* real_seg_size, 0, 0);
                            ompi_datatype_copy_content_same_ddt(datatype, send_count, con->cpu_buff_list[i], (char*)context->buff);
                            con->cpu_buff_memcpy_flags[i] = CPU_BUFFER_MEMCPY_DONE;
                        }
                        send_buff = con->cpu_buff_list[i];
                        if (con->cpu_buff_memcpy_flags[i] != CPU_BUFFER_MEMCPY_DONE) {
                            context->send_count = send_count;
                            context->buff = send_buff;
                            context->flags = COLL_ADAPT_CONTEXT_FLAGS_CUDA_BCAST;
                            context->cuda_callback = bcast_send_context_async_memcpy_callback;
                            context->debug_flag = 999;
                            mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)context);
                            continue;
                        }
                    }
                }
#endif
                OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d tag %d\n", rank, gettid(), context->frag_id, context->peer, (void *)send_buff, send_count, (ibcast_tag << 16) + i));
                err = MCA_PML_CALL(isend(send_buff, send_count, datatype, context->peer, (ibcast_tag << 16) + i, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
                if (MPI_SUCCESS != err) {
                    return err;
                }
                /* invoke send call back */
                OPAL_THREAD_UNLOCK(mutex);
                ompi_request_set_callback(send_req, send_cb, context);
                OPAL_THREAD_LOCK(mutex);
            }
        }
        
    }
    
    /* if not root, receive data from parent in the tree. */
    else{
        /* handle the situation is num_segs < RECV_NUM */
        if (num_segs <= coll_adapt_ibcast_max_recv_requests) {
            min = num_segs;
        }
        else{
            min = coll_adapt_ibcast_max_recv_requests;
        }
        
        /* set recv_array, recv_array is empty */
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = 0;
        }
        /* set send_array to empty */
        for (i = 0; i < tree->tree_nextsize ; i++) {
            send_array[i] = 0;
        }
        
        /* create a recv request */
        ompi_request_t *recv_req;
        
        /* recevice some segments from its parent */
        int recv_count = seg_count;
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                recv_count = count - i * seg_count;
            }
            mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
            context->buff = (char *)buff + i * real_seg_size;
            context->frag_id = i;
            context->peer = tree->tree_prev;
            context->con = con;
            OBJ_RETAIN(con);
            char* recv_buff = context->buff;
#if OPAL_CUDA_SUPPORT
            if (con->gpu_use_cpu_buff) {
                if (tree->topo_flags == 1 || tree->topo_flags == 0) { /* socket or node leader, receive to cpu mem */
                    if (con->cpu_buff_list == NULL) {
                        bcast_init_cpu_buff(con);
                    }
                    assert(con->cpu_buff_list != NULL);
                    con->cpu_buff_list[i] = mpool->mpool_alloc(mpool, sizeof(char)* real_seg_size, 0, 0);
                    recv_buff = con->cpu_buff_list[i];
                }
            }
#endif
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d tag %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)recv_buff, recv_count, (ibcast_tag << 16) + i));
            err = MCA_PML_CALL(irecv(recv_buff, recv_count, datatype, context->peer, (ibcast_tag << 16) + i, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            /* invoke receive call back */
            OPAL_THREAD_UNLOCK(mutex);
            ompi_request_set_callback(recv_req, recv_cb, context);
            OPAL_THREAD_LOCK(mutex);
        }
        
    }
    
    OPAL_THREAD_UNLOCK(mutex);
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: End of Ibcast\n", rank, gettid()));
    
    return MPI_SUCCESS;
}

#if OPAL_CUDA_SUPPORT
static int bcast_init_cpu_buff(mca_coll_adapt_constant_bcast_context_t *con)
{
    int k;
    con->cpu_buff_list = malloc(sizeof(char*) * con->num_segs);
    assert(con->cpu_buff_list != NULL);
    con->cpu_buff_memcpy_flags = (int *)malloc(sizeof(int) * con->num_segs);
    con->cpu_buff_list_ref_count = (int *)malloc(sizeof(int) * con->num_segs);
    for (k = 0; k < con->num_segs; k++) {
        con->cpu_buff_memcpy_flags[k] = CPU_BUFFER_MEMCPY_NOT_DONE;
        con->cpu_buff_list[k] = NULL;
        con->cpu_buff_list_ref_count[k] = 0;
    }
    return OMPI_SUCCESS;
}

static int bcast_send_context_async_memcpy_callback(mca_coll_adapt_bcast_context_t *send_context)
{
    ompi_request_t *send_req;
    if (send_context->debug_flag == 999) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "progress bcast context %p, frag id %d\n", send_context, send_context->frag_id));
    }
    send_context->con->cpu_buff_memcpy_flags[send_context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
    int err = MCA_PML_CALL(isend(send_context->buff, send_context->send_count, send_context->con->datatype, send_context->peer, (send_context->con->ibcast_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
    ompi_request_set_callback(send_req, send_cb, send_context);
    return OMPI_SUCCESS;
}

static int update_ref_count(mca_coll_adapt_bcast_context_t *context)
{
    ompi_coll_tree_t *tree = context->con->tree;
    
    mca_mpool_base_module_t *mpool = mca_coll_adapt_component.pined_cpu_mpool;
    
    if (context->con->cpu_buff_list != NULL) {
        context->con->cpu_buff_list_ref_count[context->frag_id] ++;
        if (tree->tree_nextsize == context->con->cpu_buff_list_ref_count[context->frag_id]) {
            if (context->con->cpu_buff_list[context->frag_id] != NULL) {
                mpool->mpool_free(mpool, context->con->cpu_buff_list[context->frag_id]);
                context->con->cpu_buff_list[context->frag_id] = NULL;
            }
        }
    }
    return OMPI_SUCCESS;
}
#endif



