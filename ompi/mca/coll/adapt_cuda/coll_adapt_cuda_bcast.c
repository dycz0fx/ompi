//TODO: change tag to 3 bits
//TODO: add sent part in root use sent array to acheive always send the next one
//TODO: move receve before send
#include "ompi_config.h"
#include "ompi/mca/pml/pml.h"
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_algorithms.h"
#include "coll_adapt_cuda_context.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"        //COLL_BASE_COMPUTED_SEGCOUNT
#include "opal/util/bit_ops.h"
#include "opal/sys/atomic.h"                //atomic
#include "ompi/mca/pml/ob1/pml_ob1.h"       //dump
#include "opal/datatype/opal_datatype_cuda.h"
#include "opal/datatype/opal_datatype.h"
#include "coll_adapt_cuda_mpool.h"
#include "coll_adapt_cuda_nccl.h"
#include "opal/mca/common/cuda/common_cuda.h"

#define SEND_NUM 2    //send how many fragments at once
#define RECV_NUM 3    //receive how many fragments at once
#define SEG_SIZE 1024*512   //size of a segment
#define FREE_LIST_NUM 10    //The start size of the free list
#define FREE_LIST_MAX 10000  //The max size of the free list
#define FREE_LIST_INC 10    //The incresment of the free list
#define TEST printfno

int bcast_count = 0;
ncclComm_t nccl_comm = NULL;

int coll_adapt_cuda_bcast_use_sync = 0;

static void printfno(){
    
}

#define TIMER_DATA_TYPE struct timeval
#define GET_TIME(TV)   gettimeofday( &(TV), NULL )
#define ELAPSED_TIME(TSTART, TEND)  (((TEND).tv_sec - (TSTART).tv_sec) * 1000000 + ((TEND).tv_usec - (TSTART).tv_usec))

/* call back of async cuda memcpy, used for node/socket leader for memcpy received cpu buffer to its gpu buffer */
static int bcast_send_context_async_memcpy_callback(mca_coll_adapt_cuda_bcast_context_t *send_context);

/* call back of async cuda memcpy, used for node/socket leader for update reference count, only used when there is no topo 2 mpi in his group */
static int bcast_send_context_async_memcpy_update_ref_count_callback(mca_coll_adapt_cuda_bcast_context_t *send_context);

static int update_ref_count(mca_coll_adapt_cuda_bcast_context_t *context);

static int send_cb(ompi_request_t *req)
{
    req->req_complete_cb_called = 1;
    
    mca_coll_adapt_cuda_bcast_context_t *context = (mca_coll_adapt_cuda_bcast_context_t *) req->req_complete_cb_data;
    
    int err;
    
    TEST("[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p \n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);
    
    opal_mutex_lock (context->con->mutex);
    int sent_id = context->con->send_array[context->child_id];
    int num_sent = ++(context->con->num_sent_segs);
    //has fragments in recv_array can be sent
    if (sent_id < context->con->num_recv_segs) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_array[sent_id];
        mca_coll_adapt_cuda_bcast_context_t * send_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
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
        TEST("[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d dataype %p\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *)send_context->con->datatype);
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
            opal_mutex_unlock (context->con->mutex);
            send_cb(send_req);
            opal_mutex_lock (context->con->mutex);
        }
    }
    opal_mutex_unlock (context->con->mutex);
    
    //check whether complete the request, need to signal after return the context
    if (num_sent == context->con->tree->tree_nextsize * context->con->num_segs) {
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_LOCK(&ompi_request_lock);
        ompi_request_complete(temp_req, 1);
        OPAL_THREAD_UNLOCK(&ompi_request_lock);
        TEST("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    req->req_free(&req);
    return 1;
}

//receive call back
static int recv_cb(ompi_request_t *req){
    
    req->req_complete_cb_called = 1;
    
    //get necessary info from request
    mca_coll_adapt_cuda_bcast_context_t *context = (mca_coll_adapt_cuda_bcast_context_t *) req->req_complete_cb_data;
    
    int err, i;
    
    TEST("[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);
    
    //store the frag_id to seg array
    opal_mutex_lock (context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    context->con->recv_array[num_recv_segs_t-1] = context->frag_id;
    
    int new_id = num_recv_segs_t + RECV_NUM - 1;
    //receive new segment
    if (new_id < context->con->num_segs) {
        ompi_request_t *recv_req;
        //get new context item from free list
        mca_coll_adapt_cuda_bcast_context_t * recv_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
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
        TEST("[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d datatype %p\n", ompi_comm_rank(context->con->comm), context->frag_id, context->peer, (void *)context->buff, recv_count, (void *)recv_context->con->datatype);
        MCA_PML_CALL(irecv(recv_context->buff, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
            opal_mutex_unlock (context->con->mutex);
            recv_cb(recv_req);
            opal_mutex_lock (context->con->mutex);
        }
    }
    
    //send segment to its children
    for (i = 0; i < context->con->tree->tree_nextsize; i++) {
        //if can send the segment now means the only segment need to be sent is the just arrived one
        if (num_recv_segs_t-1 == context->con->send_array[i]) {
            ompi_request_t *send_req;
            int send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            mca_coll_adapt_cuda_bcast_context_t * send_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->tree->tree_next[i];
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_array[i]);
            TEST("[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d datatype %p comm %p\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *) send_context->con->datatype, (void *) send_context->con->comm);
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                opal_mutex_unlock (context->con->mutex);
                send_cb(send_req);
                opal_mutex_lock (context->con->mutex);
            }
        }
    }
    opal_mutex_unlock (context->con->mutex);
    
    
    
    //if this is leaf and has received all the segments
    if (context->con->tree->tree_nextsize == 0 && num_recv_segs_t == context->con->num_segs) {
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_LOCK(&ompi_request_lock);
        ompi_request_complete(temp_req, 1);
        OPAL_THREAD_UNLOCK(&ompi_request_lock);
        TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    req->req_free(&req);
    return 1;
}

//send call back
static int send_cb_cpu(ompi_request_t *req)
{
    req->req_complete_cb_called = 1;
    
    mca_coll_adapt_cuda_bcast_context_t *context = (mca_coll_adapt_cuda_bcast_context_t *) req->req_complete_cb_data;
    
    ompi_coll_tree_t *tree = context->con->tree;
    
    char *send_buff = NULL;
    
    int err;
    
    mca_mpool_base_module_t *mpool = mca_coll_adapt_cuda_component.pined_cpu_mpool;
    
    TEST("[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p \n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);
    
    opal_mutex_lock (context->con->mutex);
    /*  check if cpu_buff_list can be released */
    update_ref_count(context);
    
    int sent_id = context->con->send_array[context->child_id];
    int num_sent = ++(context->con->num_sent_segs);
    //has fragments in recv_array can be sent
    if (sent_id < context->con->num_recv_segs) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_array[sent_id];
        mca_coll_adapt_cuda_bcast_context_t * send_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        send_context->frag_id = new_id;
        send_context->child_id = context->child_id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        send_context->cuda_callback = NULL;
        OBJ_RETAIN(context->con);
        
        size_t type_size;
        ompi_datatype_type_size(send_context->con->datatype, &type_size);
        
        int send_count = send_context->con->seg_count;
        if (new_id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - new_id * send_context->con->seg_count;
        }
        
        if (tree->topo_flags == 0 && tree->tree_prev_topo_flags == -1) {   /* root */
            /* send to socket or node leader */
            if (tree->tree_next_topo_flags[context->child_id] != 2) {
                assert(context->con->cpu_buff_list != NULL);
                if (context->con->cpu_buff_memcpy_flags[new_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                    context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
                   // opal_cuda_memcpy_sync(context->con->cpu_buff_list[new_id], (char*)send_context->buff, send_count*type_size);
                    ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->con->cpu_buff_list[new_id], (char*)send_context->buff);
                    context->con->cpu_buff_memcpy_flags[new_id] = CPU_BUFFER_MEMCPY_DONE;
                }
                send_buff = context->con->cpu_buff_list[new_id];
            } else {
                send_buff = send_context->buff;
            }
        } else if (tree->topo_flags == 0 && tree->tree_prev_topo_flags == 0) {  /* non-root node leader */
            assert (0);
            if (tree->tree_next_topo_flags[context->child_id] == 2) {
                assert(context->con->cpu_buff_list != NULL);
                if (context->con->cpu_buff_memcpy_flags[new_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                    opal_output(0, "topo 0 0 sendcb memcpy\n");
                    context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
                   // opal_cuda_memcpy_sync(send_context->buff, context->con->cpu_buff_list[new_id], send_count*type_size);
                    ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, send_context->buff, context->con->cpu_buff_list[new_id]);
                    context->con->cpu_buff_memcpy_flags[new_id] = CPU_BUFFER_MEMCPY_DONE;
                }
                send_buff = send_context->buff;
            } else {
                send_buff = context->con->cpu_buff_list[new_id];
            }
        } else if (tree->topo_flags == 1) {  /* socket leader */
            assert(0);
    //        opal_output(0, "topo 1 sendcb memcpy\n");
            if (context->con->cpu_buff_memcpy_flags[new_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
               // opal_cuda_memcpy_sync(send_context->buff, context->con->cpu_buff_list[new_id], send_count*type_size);
                ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, send_context->buff, context->con->cpu_buff_list[new_id]);
                context->con->cpu_buff_memcpy_flags[new_id] = CPU_BUFFER_MEMCPY_DONE;
            }
            send_buff = send_context->buff;
        } else {
            assert(0);
            send_buff = send_context->buff;
        }

        ++(send_context->con->send_array[send_context->child_id]);
        TEST("[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d dataype %p\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *)send_context->con->datatype);
        err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb_cpu, send_context)) {
            opal_mutex_unlock (context->con->mutex);
            send_cb_cpu(send_req);
            opal_mutex_lock (context->con->mutex);
        }
    }
    opal_mutex_unlock (context->con->mutex);
    
    //check whether complete the request, need to signal after return the context
    if (num_sent == context->con->tree->tree_nextsize * context->con->num_segs) {
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_LOCK(&ompi_request_lock);
        ompi_request_complete(temp_req, 1);
        OPAL_THREAD_UNLOCK(&ompi_request_lock);
        TEST("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    req->req_free(&req);
    return 1;
}

//receive call back
static int recv_cb_cpu(ompi_request_t *req){
    
    req->req_complete_cb_called = 1;
    
    //get necessary info from request
    mca_coll_adapt_cuda_bcast_context_t *context = (mca_coll_adapt_cuda_bcast_context_t *) req->req_complete_cb_data;
    
    ompi_coll_tree_t *tree = context->con->tree;
    
    int err, i;
    char *send_buff = NULL;
    char *recv_buff = NULL;
    
    mca_mpool_base_module_t *mpool = mca_coll_adapt_cuda_component.pined_cpu_mpool;
    int leaf_cuda_memcpy_done = 1;
    
    TEST("[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);
    
    //store the frag_id to seg array
    opal_mutex_lock (context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    context->con->recv_array[num_recv_segs_t-1] = context->frag_id;
    
    int new_id = num_recv_segs_t + RECV_NUM - 1;
    //receive new segment
    if (new_id < context->con->num_segs) {
        ompi_request_t *recv_req;
        //get new context item from free list
        mca_coll_adapt_cuda_bcast_context_t * recv_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
        recv_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        recv_context->cuda_callback = NULL;
        OBJ_RETAIN(context->con);
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        /* node / socket leader, receive to cpu mem */
        if ((tree->topo_flags == 1 && tree->tree_prev_topo_flags == 0) || (tree->topo_flags == 0 && tree->tree_prev_topo_flags == 0)) {
       //     opal_output(0, "recvcb change recv_buff\n");
            context->con->cpu_buff_list[new_id] = mpool->mpool_alloc(mpool, sizeof(char)* context->con->real_seg_size, 0, 0);
            recv_buff = context->con->cpu_buff_list[new_id];
        } else {
            recv_buff = recv_context->buff;
        }
        TEST("[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d datatype %p\n", ompi_comm_rank(context->con->comm), context->frag_id, context->peer, (void *)context->buff, recv_count, (void *)recv_context->con->datatype);
        MCA_PML_CALL(irecv(recv_buff, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        if(!ompi_request_set_callback(recv_req, recv_cb_cpu, recv_context)) {
            opal_mutex_unlock (context->con->mutex);
            recv_cb_cpu(recv_req);
            opal_mutex_lock (context->con->mutex);
        }
    }
    
    size_t type_size;
    ompi_datatype_type_size(context->con->datatype, &type_size);
    int send_count;
    
    //send segment to its children
    for (i = 0; i < context->con->tree->tree_nextsize; i++) {
        //if can send the segment now means the only segment need to be sent is the just arrived one
        if (num_recv_segs_t-1 == context->con->send_array[i]) {
            ompi_request_t *send_req;
            send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            mca_coll_adapt_cuda_bcast_context_t * send_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->tree->tree_next[i];
            send_context->con = context->con;
            send_context->cuda_callback = NULL;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_array[i]);
            if ((tree->topo_flags == 1 && tree->tree_prev_topo_flags == 0)) {  /* socket leader */
        //        opal_output(0, "topo 1 recv cb memcpy\n");
                assert(context->con->cpu_buff_list[context->frag_id] != NULL);
                if (context->con->cpu_buff_memcpy_flags[context->frag_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                    if (coll_adapt_cuda_bcast_use_sync) {
                    //    opal_cuda_memcpy_sync(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                        ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
                        context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
                    } else {
                        context->con->datatype->super.flags |= OPAL_DATATYPE_FLAG_GPU_ASYNC;
                      //  mca_common_cuda_memcpy_async(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                        ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
                        send_context->send_count = send_count;
                        send_context->buff = context->buff;
                        send_context->flags = COLL_ADAPT_CUDA_CONTEXT_FLAGS_BCAST;
                        send_context->cuda_callback = bcast_send_context_async_memcpy_callback;
                        mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)send_context);
                        context->con->datatype->super.flags &= ~OPAL_DATATYPE_FLAG_GPU_ASYNC;
                        context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
                      //  opal_output(0, "topo 0 recv cb record event\n");
                        continue;
                    }
                }
                send_context->buff = context->buff;
                send_buff = send_context->buff;
            } else if (tree->topo_flags == 0 && tree->tree_prev_topo_flags == 0) {  /* node leader */
                assert(context->con->cpu_buff_list[context->frag_id] != NULL);
                /* send to process in his group */
                if (tree->tree_next_topo_flags[i] == 2) {
            //              opal_output(0, "topo 0 recv cb memcpy\n");
                    if (context->con->cpu_buff_memcpy_flags[context->frag_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                        if (coll_adapt_cuda_bcast_use_sync) {
                            //opal_cuda_memcpy_sync(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                            ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
                            context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
                        } else {
                            //mca_common_cuda_memcpy_async(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                            context->con->datatype->super.flags |= OPAL_DATATYPE_FLAG_GPU_ASYNC;
                            ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
                            send_context->send_count = send_count;
                            send_context->buff = context->buff;
                            send_context->flags = COLL_ADAPT_CUDA_CONTEXT_FLAGS_BCAST;
                            send_context->cuda_callback = bcast_send_context_async_memcpy_callback;
                            mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)send_context);
                            context->con->datatype->super.flags &= ~OPAL_DATATYPE_FLAG_GPU_ASYNC;
                            context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
                          //  opal_output(0, "topo 0 recv cb record event\n");
                            continue;
                        }
                    }
                    send_context->buff = context->buff;
                    send_buff = send_context->buff;
                } else { /* node send to node/socket leader, use cpu */
    //                opal_output(0, "topo 0 recv cb direct send\n");
                    send_buff = context->con->cpu_buff_list[context->frag_id];
                }
            } else {
                send_buff = send_context->buff;
            }
            TEST("[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d datatype %p comm %p\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *) send_context->con->datatype, (void *) send_context->con->comm);
            err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb_cpu, send_context)) {
                opal_mutex_unlock (context->con->mutex);
                send_cb_cpu(send_req);
                opal_mutex_lock (context->con->mutex);
            }
        }
    }
    /* node / socket leader, if they dont have children or children topo is 1, make sure, copy data back to gpu mem */
    if ((tree->topo_flags == 1 && tree->tree_prev_topo_flags == 0) || (tree->topo_flags == 0 && tree->tree_prev_topo_flags == 0)) { 
        if (context->con->cpu_buff_memcpy_flags[context->frag_id] == CPU_BUFFER_MEMCPY_NOT_DONE) {
            send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            if (coll_adapt_cuda_bcast_use_sync) {
            //    opal_cuda_memcpy_sync(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
            } else {
                context->con->datatype->super.flags |= OPAL_DATATYPE_FLAG_GPU_ASYNC;
           //     mca_coll_adapt_cuda_bcast_context_t * cuda_memcpy_context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context->con->context_list);
            //    cuda_memcpy_context->con = context->con;
              //  cuda_memcpy_context->cuda_callback = bcast_send_context_async_memcpy_update_ref_count_callback;
               // mca_common_cuda_memcpy_async(context->buff, context->con->cpu_buff_list[context->frag_id], send_count*type_size);
                ompi_datatype_copy_content_same_ddt(context->con->datatype, send_count, context->buff, context->con->cpu_buff_list[context->frag_id]);
                context->con->datatype->super.flags &= ~OPAL_DATATYPE_FLAG_GPU_ASYNC;
             //   mca_common_cuda_record_memcpy_event("memcpy in coll_adapt_cuda_bcast", (void *)context);
            }
            context->con->cpu_buff_memcpy_flags[context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
            leaf_cuda_memcpy_done = 0;
        }
    }
    opal_mutex_unlock (context->con->mutex);
    
    
    
    //if this is leaf and has received all the segments
    if (context->con->tree->tree_nextsize == 0 && num_recv_segs_t == context->con->num_segs) {
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_LOCK(&ompi_request_lock);
        ompi_request_complete(temp_req, 1);
        OPAL_THREAD_UNLOCK(&ompi_request_lock);
        TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_cuda_bcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
  //  ncclStream_t stream = (ncclStream_t)mca_common_cuda_get_nccl_stream();
    if (opal_datatype_cuda_kernel_support == 0 || mca_common_cuda_is_stage_three_init() == 0) {
        printf("cuda pipeline\n");
        return mca_coll_adapt_cuda_bcast_pipeline(buff, count, datatype, root, comm, module);
    } else {
       // return mca_coll_adapt_cuda_bcast_pipeline(buff, count, datatype, root, comm, module);
        ncclUniqueId commId;
        int pid = 28987;
        int cid = 1;
        int len = snprintf(commId.internal, NCCL_UNIQUE_ID_BYTES, "nccl-%d-%d", pid, cid);
        int size = ompi_comm_size(comm);
        int rank = ompi_comm_rank(comm);
       // coll_adapt_cuda_nccl_comm_init_rank(&nccl_comm, size, commId, rank);
    //return mca_coll_adapt_cuda_bcast_pipeline(buff, count, datatype, root, comm, module);
        if (1 == opal_cuda_is_gpu_buffer(buff)) {
            return mca_coll_adapt_cuda_bcast_topoaware_chain(buff, count, datatype, root, comm, module);
        } else {
            return mca_coll_adapt_cuda_bcast_pipeline(buff, count, datatype, root, comm, module);
        }
    }
}

int mca_coll_adapt_cuda_bcast_nccl(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
   //  ncclUniqueId commId;
   //  int pid = 28987;
   //  int cid = 1;
   //  int len = snprintf(commId.internal, NCCL_UNIQUE_ID_BYTES, "nccl-%d-%d", pid, cid);
   //  int size = ompi_comm_size(comm);
   //  int rank = ompi_comm_rank(comm);
   //  ncclStream_t stream = (ncclStream_t)mca_common_cuda_get_nccl_stream();
   //  if (stream != NULL) {
   //      if (nccl_comm == NULL) {
   //          printf("init nccl comm late\n");
   //          coll_adapt_cuda_nccl_comm_init_rank(&nccl_comm, size, commId, rank);
   //      }
   //      int i = 0;
   //      int seg_size = SEG_SIZE;
   //      int num_segs, seg_count, send_count;
   //      size_t type_size;           //the size of a datatype
   //      size_t real_seg_size;       //the real size of a segment
   //      ptrdiff_t extent, lb;
   //      ompi_datatype_type_size(datatype, &type_size);
   //      COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
   //
   //      ompi_datatype_get_extent(datatype, &lb, &extent);
   //      num_segs = (count + seg_count - 1) / seg_count;
   //      real_seg_size = (ptrdiff_t)seg_count * extent;
   //
   //      send_count = seg_count;
   //      for (i = 0; i < num_segs; i++) {
   //          if (i == (num_segs-1)) {
   //              send_count = send_count = count - i * seg_count;
   //          }
   //          coll_adapt_cuda_nccl_bcast(buff + i*real_seg_size, send_count, ncclChar, root, nccl_comm, stream);
   //      }
   //      mca_common_cuda_sync_nccl_stream();
   // // coll_adapt_cuda_nccl_comm_destroy(nccl_comm);
   //  } else {
   //      printf("skip, stream is not ready\n");
   //  }
    return MPI_SUCCESS;
}


// broadcast using binomial tree with pipeline
int mca_coll_adapt_cuda_bcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bmtree) && (coll_comm->cached_bmtree_root == root) ) ) {
        if( coll_comm->cached_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bmtree) );
        }
        coll_comm->cached_bmtree = ompi_coll_base_topo_build_bmtree(comm, root);
        coll_comm->cached_bmtree_root = root;
    }
    //print_tree(coll_comm->cached_bmtree, ompi_comm_rank(comm));
    coll_comm->cached_bmtree->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_bmtree);
}

int mca_coll_adapt_cuda_bcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_in_order_bmtree) && (coll_comm->cached_in_order_bmtree_root == root) ) ) {
        if( coll_comm->cached_in_order_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_in_order_bmtree) );
        }
        coll_comm->cached_in_order_bmtree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
        coll_comm->cached_in_order_bmtree_root = root;
    }
    coll_comm->cached_in_order_bmtree->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_in_order_bmtree);
}

int mca_coll_adapt_cuda_bcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bintree) && (coll_comm->cached_bintree_root == root) ) ) {
        if( coll_comm->cached_bintree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bintree) );
        }
        coll_comm->cached_bintree = ompi_coll_base_topo_build_tree(2, comm, root);
        coll_comm->cached_bintree_root = root;
    }
    coll_comm->cached_bintree->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_bintree);
}

int mca_coll_adapt_cuda_bcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_pipeline) && (coll_comm->cached_pipeline_root == root) ) ) {
        if( coll_comm->cached_pipeline ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_pipeline) );
        }
        coll_comm->cached_pipeline = ompi_coll_base_topo_build_chain(1, comm, root);
        coll_comm->cached_pipeline_root = root;
    }
    coll_comm->cached_pipeline->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_pipeline);
}

int mca_coll_adapt_cuda_bcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_chain) && (coll_comm->cached_chain_root == root) ) ) {
        if( coll_comm->cached_chain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_chain) );
        }
        coll_comm->cached_chain = ompi_coll_base_topo_build_chain(4, comm, root);
        coll_comm->cached_chain_root = root;
    }
    coll_comm->cached_chain->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_chain);
}

int mca_coll_adapt_cuda_bcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    int fanout = ompi_comm_size(comm) - 1;
    ompi_coll_tree_t * tree;
    if (fanout > 1) {
        tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    }
    else{
        tree = ompi_coll_base_topo_build_chain(1, comm, root);
    }
    tree->topo_flags = -1;
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, tree);
}

int mca_coll_adapt_cuda_bcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topolinear) && (coll_comm->cached_topolinear_root == root) ) ) {
        if( coll_comm->cached_topolinear ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topolinear) );
        }
        coll_comm->cached_topolinear = ompi_coll_base_topo_build_topoaware_linear(comm, root, module);
        coll_comm->cached_topolinear_root = root;
    }
    return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_topolinear);
}

int mca_coll_adapt_cuda_bcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topochain) && (coll_comm->cached_topochain_root == root) ) ) {
        if( coll_comm->cached_topochain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topochain) );
        }
        coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module);
        coll_comm->cached_topochain_root = root;
    }
    else {
    }
    //print_tree(coll_comm->cached_topochain, ompi_comm_rank(comm));
    return mca_coll_adapt_cuda_bcast_generic_cpu(buff, count, datatype, root, comm, module, coll_comm->cached_topochain);
    //return mca_coll_adapt_cuda_bcast_generic(buff, count, datatype, root, comm, module, coll_comm->cached_topochain);
}

static int print_topo_level(int rank, ompi_coll_tree_t* tree)
{
    printf("rank %d, pid %d, topo_level %d, parent [%d topo %d], nb child %d, ", rank, getpid(), tree->topo_flags, tree->tree_prev, tree->tree_prev_topo_flags, tree->tree_nextsize);
    int i;
    for (i=0; i<tree->tree_nextsize; i++) {
        printf("child [%d, topo %d], ", tree->tree_next[i], tree->tree_next_topo_flags[i]);
    }
    printf("\n");
    return 0;
}

int mca_coll_adapt_cuda_bcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    int i, j;       //temp variable for iteration
    int size;       //size of the communicator
    int rank;       //rank of this node
    int err;        //record return value
    int min;        //the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM
    
    size_t seg_size;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    size_t type_size;           //the size of a datatype
    size_t real_seg_size;       //the real size of a segment
    ptrdiff_t extent, lb;
    int num_segs;               //the number of segments
    
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    opal_mutex_t * mutex;
    int *recv_array = NULL;   //store those segments which are received
    int *send_array = NULL;   //record how many isend has been issued for every child
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_cuda_bcast_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_cuda_bcast_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM,
                        FREE_LIST_MAX,
                        FREE_LIST_INC,
                        NULL, 0, NULL, NULL, NULL);
    
    
    seg_size = SEG_SIZE;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    //Determine number of elements sent per operation
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    
    ompi_datatype_get_extent(datatype, &lb, &extent);
    num_segs = (count + seg_count - 1) / seg_count;
    real_seg_size = (ptrdiff_t)seg_count * extent;
    
    //set memory for recv_array and send_array, created on heap becasue they are needed to be accessed by other functions, callback function
    if (num_segs!=0) {
        recv_array = (int *)malloc(sizeof(int) * num_segs);
    }
    if (tree->tree_nextsize!=0) {
        send_array = (int *)malloc(sizeof(int) * tree->tree_nextsize);
    }
    
    //set up mutex
    mutex = OBJ_NEW(opal_mutex_t);
    
    ompi_request_t * temp_request = NULL;
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    
    //Set constant context for send and recv call back
    mca_coll_adapt_cuda_constant_bcast_context_t *con = OBJ_NEW(mca_coll_adapt_cuda_constant_bcast_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = datatype;
    con->comm = comm;
    con->real_seg_size = real_seg_size;
    con->num_segs = num_segs;
    con->request = temp_request;
    con->mutex = mutex;
    con->context_list = context_list;
    con->recv_array = recv_array;
    con->num_recv_segs = 0;
    con->send_array = send_array;
    con->num_sent_segs = 0;
    con->tree = tree;
    
    TEST("[%d, %" PRIx64 "]: Bcast, root %d\n", rank, gettid(), root);
    TEST("[%d, %" PRIx64 "]: con->mutex = %p, num_children = %d\n", rank, gettid(), (void *)con->mutex, tree->tree_nextsize);
    //if root, send segment to every children.
    
    opal_mutex_lock(mutex);
    print_topo_level(rank, tree);
    
    if (rank == root){
        //handle the situation when num_segs < SEND_NUM
        if (num_segs <= SEND_NUM) {
            min = num_segs;
        }
        else{
            min = SEND_NUM;
        }
        
        //set recv_array and num_recv_segs, root has already had all the segments
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = i;
        }
        con->num_recv_segs = num_segs;
        //set send_array, has not sent any segments
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = min;
        }
        
        ompi_request_t *send_req;
        int send_count = seg_count;             //number of datatype in each send
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                send_count = count - i * seg_count;
            }
            for (j=0; j<tree->tree_nextsize; j++) {
                mca_coll_adapt_cuda_bcast_context_t * context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context_list);
                context->buff = (char *)buff + i * real_seg_size;
                context->frag_id = i;
                context->child_id = j;              //the id of peer in in tree->tree_next
                context->peer = tree->tree_next[j];   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                TEST("[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d datatype %p\n", rank, gettid(), context->frag_id, context->peer, (void *)context->buff, send_count, (void *)datatype);
                err = MCA_PML_CALL(isend(context->buff, send_count, datatype, context->peer, i, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
                
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //invoke send call back
                if(!ompi_request_set_callback(send_req, send_cb, context)) {
                    opal_mutex_unlock(mutex);
                    send_cb(send_req);
                    opal_mutex_lock(mutex);
                }
                
            }
        }
        
        if (tree->tree_nextsize != 0) {
            opal_mutex_unlock(mutex);
            ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
        }
        
    }
    
    //if not root, receive data from parent in the tree.
    else {
        //handle the situation is num_segs < RECV_NUM
        if (num_segs <= RECV_NUM) {
            min = num_segs;
        }
        else{
            min = RECV_NUM;
        }
        
        //set recv_array, recv_array is empty and num_recv_segs is 0
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = 0;
        }
        //set send_array to empty
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = 0;
        }
        
        //create a recv request
        ompi_request_t *recv_req;
        
        //recevice some segments from its parent
        int recv_count = seg_count;
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                recv_count = count - i * seg_count;
            }
            mca_coll_adapt_cuda_bcast_context_t * context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context_list);
            
            context->buff = (char *)buff + i * real_seg_size;
            context->frag_id = i;
            context->peer = tree->tree_prev;
            context->con = con;
            OBJ_RETAIN(con);
            TEST("[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d datatype %p comm %p\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, recv_count, (void *)datatype, (void *)comm);
            err = MCA_PML_CALL(irecv(context->buff, recv_count, datatype, context->peer, context->frag_id, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke receive call back
            if(!ompi_request_set_callback(recv_req, recv_cb, context)) {
                opal_mutex_unlock(mutex);
                recv_cb(recv_req);
                opal_mutex_lock(mutex);
                
            }
        }
        
        opal_mutex_unlock(mutex);
        ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
        
    }
    
    
    TEST("[%d, %" PRIx64 "]: End of bcast\n", rank, gettid());
    
    if (tree->tree_nextsize != 0) {
        free(con->send_array);
    }
    if (con->num_segs !=0) {
        free(con->recv_array);
    }
    OBJ_RELEASE(con->mutex);
    OBJ_RELEASE(con);
    OBJ_RELEASE(context_list);
    return MPI_SUCCESS;
}

int mca_coll_adapt_cuda_bcast_generic_cpu(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    int i, j, k;       //temp variable for iteration
    int size;       //size of the communicator
    int rank;       //rank of this node
    int err;        //record return value
    int min;        //the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM
    
    size_t seg_size;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    size_t type_size;           //the size of a datatype
    size_t real_seg_size;       //the real size of a segment
    ptrdiff_t extent, lb;
    int num_segs;               //the number of segments
    
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    opal_mutex_t * mutex;
    int *recv_array = NULL;   //store those segments which are received
    int *send_array = NULL;   //record how many isend has been issued for every child
    
    opal_free_list_t *cpu_buff_list = NULL;  // used to send/receive data into cpu mem, only used for node leader
    char *send_buff = NULL;
    char *recv_buff = NULL;
    mca_mpool_base_module_t *mpool = mca_coll_adapt_cuda_component.pined_cpu_mpool;
    
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_cuda_bcast_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_cuda_bcast_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM,
                        FREE_LIST_MAX,
                        FREE_LIST_INC,
                        NULL, 0, NULL, NULL, NULL);
    
    
    seg_size = SEG_SIZE;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    if (rank == root)  printf("cuda topo chain cpu\n");
    
    
    //Determine number of elements sent per operation
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    
    ompi_datatype_get_extent(datatype, &lb, &extent);
    num_segs = (count + seg_count - 1) / seg_count;
    real_seg_size = (ptrdiff_t)seg_count * extent;
    
    //set memory for recv_array and send_array, created on heap becasue they are needed to be accessed by other functions, callback function
    if (num_segs!=0) {
        recv_array = (int *)malloc(sizeof(int) * num_segs);
    }
    if (tree->tree_nextsize!=0) {
        send_array = (int *)malloc(sizeof(int) * tree->tree_nextsize);
    }
    
    //set up mutex
    mutex = OBJ_NEW(opal_mutex_t);
    
    ompi_request_t * temp_request = NULL;
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    
    //Set constant context for send and recv call back
    mca_coll_adapt_cuda_constant_bcast_context_t *con = OBJ_NEW(mca_coll_adapt_cuda_constant_bcast_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = datatype;
    con->comm = comm;
    con->real_seg_size = real_seg_size;
    con->num_segs = num_segs;
    con->request = temp_request;
    con->mutex = mutex;
    con->context_list = context_list;
    con->recv_array = recv_array;
    con->num_recv_segs = 0;
    con->send_array = send_array;
    con->num_sent_segs = 0;
    con->tree = tree;
    con->cpu_buff_list = NULL;
    con->cpu_buff_memcpy_flags = NULL;
    con->cpu_buff_list_ref_count = NULL;
    
    TEST("[%d, %" PRIx64 "]: Bcast, root %d\n", rank, gettid(), root);
    TEST("[%d, %" PRIx64 "]: con->mutex = %p, num_children = %d\n", rank, gettid(), (void *)con->mutex, tree->tree_nextsize);
    //if root, send segment to every children.
    
    opal_mutex_lock(mutex);
    
    print_topo_level(rank, tree);
    
    if (rank == root){
        
        tree->tree_prev_topo_flags = -1;
        //handle the situation when num_segs < SEND_NUM
        if (num_segs <= SEND_NUM) {
            min = num_segs;
        }
        else{
            min = SEND_NUM;
        }
        
        //set recv_array and num_recv_segs, root has already had all the segments
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = i;
        }
        con->num_recv_segs = num_segs;
        //set send_array, has not sent any segments
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = min;
        }
        
        ompi_request_t *send_req;
        int send_count = seg_count;             //number of datatype in each send
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                send_count = count - i * seg_count;
            }
            for (j=0; j<tree->tree_nextsize; j++) {
                mca_coll_adapt_cuda_bcast_context_t * context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context_list);
                context->buff = (char *)buff + i * real_seg_size;
                context->frag_id = i;
                context->child_id = j;              //the id of peer in in tree->tree_next
                context->peer = tree->tree_next[j];   //the actural rank of the peer
                context->con = con;
                context->cuda_callback = NULL;
                OBJ_RETAIN(con);
                
                /* socket or node leader, send through cpu mem */
                if (tree->topo_flags != -1 && tree->tree_next_topo_flags[j] != 2) {
                    if (con->cpu_buff_list == NULL) {
                        con->cpu_buff_list = malloc(sizeof(char*) * num_segs);
                        assert(con->cpu_buff_list != NULL);
                        con->cpu_buff_memcpy_flags = (int *)malloc(sizeof(int) * num_segs);
                        con->cpu_buff_list_ref_count = (int *)malloc(sizeof(int) * num_segs);
                        for (k = 0; k < num_segs; k++) {
                            con->cpu_buff_memcpy_flags[k] = CPU_BUFFER_MEMCPY_NOT_DONE;
                            con->cpu_buff_list[k] = NULL;
                            con->cpu_buff_list_ref_count[k] = 0;
                        }
                    }
                    if (con->cpu_buff_memcpy_flags[i] == CPU_BUFFER_MEMCPY_NOT_DONE) {
                       // ompi_datatype_copy_content_same_ddt(datatype, send_count, (char*)cpu_buff_list + i * real_seg_size, (char*)context->buff);
                        con->cpu_buff_list[i] = mpool->mpool_alloc(mpool, sizeof(char)* real_seg_size, 0, 0);
                      //  opal_cuda_memcpy_sync(con->cpu_buff_list[i], (char*)context->buff, send_count*type_size);
                        ompi_datatype_copy_content_same_ddt(datatype, send_count, con->cpu_buff_list[i], (char*)context->buff);
                        con->cpu_buff_memcpy_flags[i] = CPU_BUFFER_MEMCPY_DONE;
                    }
                    send_buff = con->cpu_buff_list[i];
                } else {
                    send_buff = context->buff;
                }
                
                TEST("[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d datatype %p\n", rank, gettid(), context->frag_id, context->peer, (void *)context->buff, send_count, (void *)datatype);
                err = MCA_PML_CALL(isend(send_buff, send_count, datatype, context->peer, i, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
                
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //invoke send call back
                if(!ompi_request_set_callback(send_req, send_cb_cpu, context)) {
                    opal_mutex_unlock(mutex);
                    send_cb_cpu(send_req);
                    opal_mutex_lock(mutex);
                }
                
            }
        }
        
        if (tree->tree_nextsize != 0) {
            opal_mutex_unlock(mutex);
            ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
        }
        
    }
    
    //if not root, receive data from parent in the tree.
    else {
        //handle the situation is num_segs < RECV_NUM
        if (num_segs <= RECV_NUM) {
            min = num_segs;
        }
        else{
            min = RECV_NUM;
        }
        
        //set recv_array, recv_array is empty and num_recv_segs is 0
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = 0;
        }
        //set send_array to empty
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = 0;
        }
        
        //create a recv request
        ompi_request_t *recv_req;
        
        //recevice some segments from its parent
        int recv_count = seg_count;
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                recv_count = count - i * seg_count;
            }
            mca_coll_adapt_cuda_bcast_context_t * context = (mca_coll_adapt_cuda_bcast_context_t *) opal_free_list_wait(context_list);
            
            context->buff = (char *)buff + i * real_seg_size;
            context->frag_id = i;
            context->peer = tree->tree_prev;
            context->con = con;
            context->cuda_callback = NULL;
            OBJ_RETAIN(con);
            
            /* socket or node leader, receive to cpu mem */
            if ( (tree->topo_flags == 1 && tree->tree_prev_topo_flags == 0) || (tree->topo_flags == 0 && tree->tree_prev_topo_flags == 0)) {
                if (con->cpu_buff_list == NULL) {
                    con->cpu_buff_list = malloc(sizeof(char*) * num_segs);
                    con->cpu_buff_memcpy_flags = (int *)malloc(sizeof(int) * num_segs);
                    con->cpu_buff_list_ref_count = (int *)malloc(sizeof(int) * num_segs);
                    for (k = 0; k < num_segs; k++) {
                        con->cpu_buff_memcpy_flags[k] = CPU_BUFFER_MEMCPY_NOT_DONE;
                        con->cpu_buff_list[k] = NULL;
                        con->cpu_buff_list_ref_count[k] = 0;
                    }
                }
                assert(con->cpu_buff_list != NULL);
                con->cpu_buff_list[i] = mpool->mpool_alloc(mpool, sizeof(char)* real_seg_size, 0, 0);
                //opal_output(0, "recv change recv buff\n");
                recv_buff = con->cpu_buff_list[i];
            } else {
                recv_buff = context->buff;
            }
            
            TEST("[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d datatype %p comm %p\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, recv_count, (void *)datatype, (void *)comm);
            err = MCA_PML_CALL(irecv(recv_buff, recv_count, datatype, context->peer, context->frag_id, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke receive call back
            if(!ompi_request_set_callback(recv_req, recv_cb_cpu, context)) {
                opal_mutex_unlock(mutex);
                recv_cb_cpu(recv_req);
                opal_mutex_lock(mutex);
                
            }
        }
        
        opal_mutex_unlock(mutex);
        ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
        
    }
    
    mca_common_cuda_sync_memcpy_stream();
    
    
    TEST("[%d, %" PRIx64 "]: End of bcast\n", rank, gettid());
    
    if (tree->tree_nextsize != 0) {
        free(con->send_array);
    }
    if (con->num_segs !=0) {
        free(con->recv_array);
    }
    int free_count = 0;
    if (con->cpu_buff_list != NULL) {
        for (k = 0; k < num_segs; k++) {
            if (con->cpu_buff_list[k] != NULL) {
                mpool->mpool_free(mpool, con->cpu_buff_list[k]);
                free_count ++;
            }
        }
        opal_output(0, "rank %d freed %d block at last\n", rank, free_count);
        free(con->cpu_buff_list);
        con->cpu_buff_list = NULL;
    }
    if (con->cpu_buff_memcpy_flags != NULL) {
        free(con->cpu_buff_memcpy_flags);
    }
    if (con->cpu_buff_list_ref_count != NULL) {
        free(con->cpu_buff_list_ref_count);
    }
    OBJ_RELEASE(con->mutex);
    OBJ_RELEASE(con);
    OBJ_RELEASE(context_list);
    return MPI_SUCCESS;
}

static int update_ref_count(mca_coll_adapt_cuda_bcast_context_t *context)
{
    ompi_coll_tree_t *tree = context->con->tree;
    
    mca_mpool_base_module_t *mpool = mca_coll_adapt_cuda_component.pined_cpu_mpool;
    
    if (context->con->cpu_buff_list != NULL) {
        context->con->cpu_buff_list_ref_count[context->frag_id] ++;
        if (tree->tree_nextsize == context->con->cpu_buff_list_ref_count[context->frag_id]) {
            if (context->con->cpu_buff_list[context->frag_id] != NULL) {
                mpool->mpool_free(mpool, context->con->cpu_buff_list[context->frag_id]);
                context->con->cpu_buff_list[context->frag_id] = NULL;
            }
        } 
    }
}

static int bcast_send_context_async_memcpy_callback(mca_coll_adapt_cuda_bcast_context_t *send_context)
{
    ompi_request_t *send_req;
  //  printf("progress bcast context %p\n", send_context);
    send_context->con->cpu_buff_memcpy_flags[send_context->frag_id] = CPU_BUFFER_MEMCPY_DONE;
    int err = MCA_PML_CALL(isend(send_context->buff, send_context->send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
    if(!ompi_request_set_callback(send_req, send_cb_cpu, send_context)) {
        send_cb_cpu(send_req);
    }
    return OMPI_SUCCESS;
}

static int bcast_send_context_async_memcpy_update_ref_count_callback(mca_coll_adapt_cuda_bcast_context_t *send_context)
{
    update_ref_count(send_context);
    return OMPI_SUCCESS;
}

int coll_adapt_cuda_bcast_progress()
{
 //   printf("i am in adapt cuda progress\n");
    char *context;
    while (1 == progress_one_cuda_memcpy_event((void **)&context)) {
        if (context != NULL) {
            int *flag = (int *)(context + sizeof(opal_free_list_item_t));
            if (*flag == COLL_ADAPT_CUDA_CONTEXT_FLAGS_BCAST) {
                mca_coll_adapt_cuda_bcast_context_t *bcast_context = (mca_coll_adapt_cuda_bcast_context_t *)context;
                assert(bcast_context->cuda_callback != NULL);
                bcast_context->cuda_callback(bcast_context);
            }
        }
    }
    return OMPI_SUCCESS;
}


