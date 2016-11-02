#include "ompi_config.h"
#include "ompi/communicator/communicator.h"
#include "coll_adapt_cuda_algorithms.h"
#include "coll_adapt_cuda_context.h"
#include "coll_adapt_cuda_item.h"
#include "coll_adapt_cuda.h"
#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "ompi/mca/coll/base/coll_base_topo.h"  //build tree
#include "opal/datatype/opal_datatype_cuda.h"
#include "coll_adapt_cuda_mpool.h"

#define SEND_NUM 2    //send how many fragments at once
#define RECV_NUM 3    //receive how many fragments at once
#define SEG_SIZE 1024*1024   //size of a segment
#define FREE_LIST_NUM_CONTEXT_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_CONTEXT_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_CONTEXT_LIST 10    //The incresment of the context free list
#define FREE_LIST_NUM_INBUF_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_INBUF_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_INBUF_LIST 10    //The incresment of the context free list
#define TEST printfno


int coll_adapt_cuda_reduce_use_sync = 0;

//Can only work on commutative op

static void printfno(){
    
}

//static size_t real_seg_size_cuda = 0;
#define TIMER_DATA_TYPE struct timeval
#define GET_TIME(TV)   gettimeofday( &(TV), NULL )
#define ELAPSED_TIME(TSTART, TEND)  (((TEND).tv_sec - (TSTART).tv_sec) * 1000000 + ((TEND).tv_usec - (TSTART).tv_usec))


static mca_coll_adapt_cuda_item_t * get_next_ready_item(opal_list_t* list, int num_children){
    mca_coll_adapt_cuda_item_t *item;
    if (opal_list_is_empty(list)) {
        return NULL;
    }
    for(item = (mca_coll_adapt_cuda_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_cuda_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_cuda_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->count == num_children) {
            opal_list_remove_item(list, (opal_list_item_t *)item);
            return item;
        }
    }
    return NULL;
}

static int add_to_list(opal_list_t* list, int id){
    mca_coll_adapt_cuda_item_t *item;
    int ret = 0;
    for(item = (mca_coll_adapt_cuda_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_cuda_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_cuda_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->id == id) {
            (item->count)++;
            ret = 1;
            break;
        }
    }
    if (ret == 0) {
        item = OBJ_NEW(mca_coll_adapt_cuda_item_t);
        item->id = id;
        item->count = 1;
        opal_list_append(list, (opal_list_item_t *)item);
        ret = 2;
    }
    TEST("add_to_list_return %d\n", ret);
    return ret;
}

#if 0

static int send_cb(ompi_request_t *req){
    mca_coll_adapt_cuda_reduce_context_t *context = (mca_coll_adapt_cuda_reduce_context_t *) req->req_complete_cb_data;
    TEST("[%d]: send_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id);
    int err;
    
    int32_t num_sent = opal_atomic_add_32(&(context->con->num_sent_segs), 1);
    opal_atomic_sub_32(&(context->con->ongoing_send), 1);
    
    //send a new segment
    //list is not empty
    opal_mutex_lock (context->con->mutex_recv_list);
    mca_coll_adapt_cuda_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    if (item != NULL) {
        //get new context item from free list
        mca_coll_adapt_cuda_reduce_context_t * send_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        if (context->con->tree->tree_nextsize > 0) {
            send_context->buff = context->con->accumbuf[item->id] - context->con->lower_bound;

        }
        else{
            send_context->buff = context->buff + (item->id - context->frag_id) * context->con->segment_increment;
        }
        send_context->frag_id = item->id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        
        opal_atomic_add_32(&(context->con->ongoing_send), 1);
        
        int send_count = send_context->con->seg_count;
        if (item->id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - item->id * send_context->con->seg_count;
        }
        
        TEST("[%d]: In send_cb, create isend to seg %d, peer %d\n", send_context->con->rank, send_context->frag_id, send_context->peer);
        
        ompi_request_t *send_req;
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        
        //release the item
        OBJ_RELEASE(item);
        
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
            send_cb(send_req);
        }
        
    }
    
    TEST("[%d]: In send_cb, root = %d, num_sent = %d, num_segs = %d\n", context->con->rank, context->con->tree->tree_root, num_sent, context->con->num_segs);
    //check whether signal the condition, non root and sent all the segments
    if (context->con->tree->tree_root != context->con->rank && num_sent == context->con->num_segs) {
        TEST("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        ompi_request_complete(temp_req, 1);
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    return MPI_SUCCESS;
}

static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_cuda_reduce_context_t *context = (mca_coll_adapt_cuda_reduce_context_t *) req->req_complete_cb_data;
    TEST("[%d]: recv_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id);
    
    int err;
    //atomic
    int32_t new_id = opal_atomic_add_32(&(context->con->next_recv_segs[context->child_id]), 1);
    
    //receive new segment
    if (new_id < context->con->num_segs) {
        //get inbuf
        char *inbuf = NULL;
        if (context->con->tree->tree_root == context->con->rank) {
            inbuf = (char*)context->con->accumbuf[new_id];
        } else {
            inbuf = (char*)opal_cuda_malloc_gpu_buffer(real_seg_size_cuda, 0);
        }
        //get new context item from free list
        mca_coll_adapt_cuda_reduce_context_t * recv_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        recv_context->inbuf = (mca_coll_adapt_cuda_inbuf_t *)inbuf;
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        
        TEST("[%d]: In recv_cb, create irecv for seg %d, peer %d\n", context->con->rank, recv_context->frag_id, recv_context->peer);
        
        ompi_request_t *recv_req;
        MCA_PML_CALL(irecv(recv_context->inbuf - recv_context->con->lower_bound, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
            recv_cb(recv_req);
        }
        
    }
    
    //do the op
    int op_count = context->con->seg_count;
    if (context->frag_id == (context->con->num_segs - 1)) {
        op_count = context->con->count - context->frag_id * context->con->seg_count;
    }
    
    int keep_inbuf = 0;
    opal_mutex_lock(context->con->mutex_op_list[context->frag_id]);
    if (context->con->accumbuf[context->frag_id] == NULL) {
        keep_inbuf = 1;
        context->con->accumbuf[context->frag_id] = context->inbuf;
        opal_cuda_recude_op_sum_double(context->con->sbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment, context->con->accumbuf[context->frag_id] - context->con->lower_bound, op_count, NULL);

    } else if (context->con->tree->tree_root == context->con->rank){
        opal_cuda_recude_op_sum_double(context->con->sbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment, context->con->accumbuf[context->frag_id] - context->con->lower_bound, op_count, NULL);
    } else {
        opal_cuda_recude_op_sum_double(context->inbuf - context->con->lower_bound, context->con->accumbuf[context->frag_id] - context->con->lower_bound, op_count, NULL);

    }

    opal_mutex_unlock(context->con->mutex_op_list[context->frag_id]);
    
    //set recv list
    opal_mutex_lock (context->con->mutex_recv_list);
    add_to_list(context->con->recv_list, context->frag_id);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    //send to parent
    if (context->con->rank != context->con->tree->tree_root && context->con->ongoing_send < SEND_NUM) {
        //atomic
        opal_mutex_lock (context->con->mutex_recv_list);
        mca_coll_adapt_cuda_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
        opal_mutex_unlock (context->con->mutex_recv_list);
        
        if (item != NULL) {
            //get new context item from free list
            mca_coll_adapt_cuda_reduce_context_t * send_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->con->accumbuf[context->frag_id] - context->con->lower_bound;
            send_context->frag_id = item->id;
            send_context->peer = context->con->tree->tree_prev;
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            //atomic
            opal_atomic_add_32(&(context->con->ongoing_send), 1);
            
            int send_count = send_context->con->seg_count;
            if (item->id == (send_context->con->num_segs - 1)) {
                send_count = send_context->con->count - item->id * send_context->con->seg_count;
            }
            
            TEST("[%d]: In recv_cb, create isend to seg %d, peer %d\n", send_context->con->rank, send_context->frag_id, send_context->peer);
            
            ompi_request_t *send_req;
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //release the item
            OBJ_RELEASE(item);
            
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                send_cb(send_req);
            }
            
        }
    }
    
    opal_mutex_lock (context->con->mutex_num_recv_segs);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    TEST("[%d]: In recv_cb, root = %d, num_recv = %d, num_segs = %d, num_child = %d\n", context->con->rank, context->con->tree->tree_root, num_recv_segs_t, context->con->num_segs, context->con->tree->tree_nextsize);
    //if this is root and has received all the segments
    if (context->con->tree->tree_root == context->con->rank && num_recv_segs_t == context->con->num_segs * context->con->tree->tree_nextsize) {
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
        ompi_request_t *temp_req = context->con->request;
        if (!keep_inbuf) {
            context->inbuf = NULL;
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        ompi_request_complete(temp_req, 1);
    } else{
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        if (!keep_inbuf) {
            TEST("return inbuf\n");
            if (context->con->tree->tree_root != context->con->rank) {
                if (context->inbuf != NULL) opal_cuda_free_gpu_buffer((void*)context->inbuf, 0);
            }
            context->inbuf = NULL;
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        
    }
    
    return MPI_SUCCESS;
}

#endif

int mca_coll_adapt_cuda_reduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    //return mca_coll_adapt_cuda_reduce_pipeline(sbuf, rbuf, count, dtype, op, root, comm, module);
     return mca_coll_adapt_cuda_reduce_topoaware_chain(sbuf, rbuf, count, dtype, op, root, comm, module);
}

int mca_coll_adapt_cuda_reduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_bmtree(comm, root);
    int r = mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_tree(2, comm, root);
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int t_count = 0;

int mca_coll_adapt_cuda_reduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    
    if(t_count++ == 0){
        TEST("Adapt reduce pipeline\n");
    }
    
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(1, comm, root);
   // int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    int r = mca_coll_adapt_cuda_reduce_chain_pipeline(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(4, comm, root);
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    //TODO: has problem when comm_size = 2
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_topoaware_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_topoaware_linear(comm, root, module);
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
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
    int r =  mca_coll_adapt_cuda_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, coll_comm->cached_topochain);
    int segcount = count;
    size_t typelng;
    ompi_datatype_type_size( dtype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( SEG_SIZE, typelng, segcount );
  // int r = mca_coll_adapt_cuda_reduce_topo_generic(sbuf, rbuf, count, dtype, op, root, comm, module, coll_comm->cached_topochain, segcount, 0);
    //int r = mca_coll_adapt_cuda_reduce_topo_generic_cpu(sbuf, rbuf, count, dtype, op, root, comm, module, coll_comm->cached_topochain, segcount, 0);
  //  ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_cuda_reduce_topo_generic( const void* sendbuf, void* recvbuf, int original_count,
                                    struct ompi_datatype_t* datatype, struct ompi_op_t* op,
                                    int root, struct ompi_communicator_t* comm,
                                    mca_coll_base_module_t *module,
                                    ompi_coll_tree_t* tree, int count_by_segment,
                                    int max_outstanding_reqs )
{
    char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
    char *accumbuf = NULL, *accumbuf_free = NULL;
    char *local_op_buffer = NULL, *sendtmpbuf = NULL;
    ptrdiff_t extent, size, gap, segment_increment;
    ompi_request_t **sreq = NULL, *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int num_segments, line, ret, segindex, i, rank;
    int recvcount, prevcount, inbi;

    /**
     * Determine number of segments and number of elements
     * sent per operation
     */
    ompi_datatype_type_extent( datatype, &extent );
    num_segments = (int)(((size_t)original_count + (size_t)count_by_segment - (size_t)1) / (size_t)count_by_segment);
    segment_increment = (ptrdiff_t)count_by_segment * extent;

    sendtmpbuf = (char*) sendbuf;
    if( sendbuf == MPI_IN_PLACE ) {
        sendtmpbuf = (char *)recvbuf;
    }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:reduce_generic count %d, msg size %ld, segsize %ld, max_requests %d",
                 original_count, (unsigned long)((ptrdiff_t)num_segments * (ptrdiff_t)segment_increment),
                 (unsigned long)segment_increment, max_outstanding_reqs));

    rank = ompi_comm_rank(comm);

    /* non-leaf nodes - wait for children to send me data & forward up
       (if needed) */
    printf("rank %d, child size %d\n", rank, tree->tree_nextsize);
    if( tree->tree_nextsize > 0 ) {
        ptrdiff_t real_segment_size;

        /* handle non existant recv buffer (i.e. its NULL) and
           protect the recv buffer on non-root nodes */
        accumbuf = (char*)recvbuf;
        if( (NULL == accumbuf) || (root != rank) ) {
            /* Allocate temporary accumulator buffer. */
            size = opal_datatype_span(&datatype->super, original_count, &gap);
            accumbuf_free = (char*)opal_cuda_malloc_gpu_buffer(size, 0);
            if (accumbuf_free == NULL) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            accumbuf = accumbuf_free - gap;
        }

        /* If this is a non-commutative operation we must copy
           sendbuf to the accumbuf, in order to simplfy the loops */
        if (!ompi_op_is_commute(op)) {
            assert(0);
            ompi_datatype_copy_content_same_ddt(datatype, original_count,
                                                (char*)accumbuf,
                                                (char*)sendtmpbuf);
        }
        /* Allocate two buffers for incoming segments */
        real_segment_size = opal_datatype_span(&datatype->super, count_by_segment, &gap);
        inbuf_free[0] = (char*)opal_cuda_malloc_gpu_buffer(real_segment_size, 0);
        if( inbuf_free[0] == NULL ) {
            line = __LINE__; ret = -1; goto error_hndl;
        }
        inbuf[0] = inbuf_free[0] - gap;
        /* if there is chance to overlap communication -
           allocate second buffer */
        if( (num_segments > 1) || (tree->tree_nextsize > 1) ) {
            inbuf_free[1] = (char*)opal_cuda_malloc_gpu_buffer(real_segment_size, 0);
            if( inbuf_free[1] == NULL ) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            inbuf[1] = inbuf_free[1] - gap;
        }

        /* reset input buffer index and receive count */
        inbi = 0;
        recvcount = 0;
        /* for each segment */
        for( segindex = 0; segindex <= num_segments; segindex++ ) {
            prevcount = recvcount;
            /* recvcount - number of elements in current segment */
            recvcount = count_by_segment;
            if( segindex == (num_segments-1) )
                recvcount = original_count - (ptrdiff_t)count_by_segment * (ptrdiff_t)segindex;

            /* for each child */
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                /**
                 * We try to overlap communication:
                 * either with next segment or with the next child
                 */
                /* post irecv for current segindex on current child */
                if( segindex < num_segments ) {
                    void* local_recvbuf = inbuf[inbi];
                    if( 0 == i ) {
                        /* for the first step (1st child per segment) and
                         * commutative operations we might be able to irecv
                         * directly into the accumulate buffer so that we can
                         * reduce(op) this with our sendbuf in one step as
                         * ompi_op_reduce only has two buffer pointers,
                         * this avoids an extra memory copy.
                         *
                         * BUT if the operation is non-commutative or
                         * we are root and are USING MPI_IN_PLACE this is wrong!
                         */
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_recvbuf = accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }

                    ret = MCA_PML_CALL(irecv(local_recvbuf, recvcount, datatype,
                                             tree->tree_next[i],
                                             MCA_COLL_BASE_TAG_REDUCE, comm,
                                             &reqs[inbi]));
                    if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;}
                }
                /* wait for previous req to complete, if any.
                   if there are no requests reqs[inbi ^1] will be
                   MPI_REQUEST_NULL. */
                /* wait on data from last child for previous segment */
                ret = ompi_request_wait_all( 1, &reqs[inbi ^ 1],
                                             MPI_STATUSES_IGNORE );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                local_op_buffer = inbuf[inbi ^ 1];
                if( i > 0 ) {
                    /* our first operation is to combine our own [sendbuf] data
                     * with the data we recvd from down stream (but only
                     * the operation is commutative and if we are not root and
                     * not using MPI_IN_PLACE)
                     */
                    if( 1 == i ) {
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }
                    /* apply operation */
                    opal_cuda_recude_op_sum_double(local_op_buffer, accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment, recvcount, NULL);
                } else if ( segindex > 0 ) {
                    void* accumulator = accumbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                    if( tree->tree_nextsize <= 1 ) {
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                        }
                    }
                    opal_cuda_recude_op_sum_double(local_op_buffer, accumulator, prevcount, NULL);

                    /* all reduced on available data this step (i) complete,
                     * pass to the next process unless you are the root.
                     */
                    if (rank != tree->tree_root) {
                        /* send combined/accumulated data to parent */
                        ret = MCA_PML_CALL( send( accumulator, prevcount,
                                                  datatype, tree->tree_prev,
                                                  MCA_COLL_BASE_TAG_REDUCE,
                                                  MCA_PML_BASE_SEND_STANDARD,
                                                  comm) );
                        if (ret != MPI_SUCCESS) {
                            line = __LINE__; goto error_hndl;
                        }
                    }

                    /* we stop when segindex = number of segments
                       (i.e. we do num_segment+1 steps for pipelining */
                    if (segindex == num_segments) break;
                }

                /* update input buffer index */
                inbi = inbi ^ 1;
            } /* end of for each child */
        } /* end of for each segment */

        /* clean up */
        if( inbuf_free[0] != NULL) { opal_cuda_free_gpu_buffer(inbuf_free[0], 0); inbuf_free[0] = NULL; }
        if( inbuf_free[1] != NULL) { opal_cuda_free_gpu_buffer(inbuf_free[1], 0); inbuf_free[1] = NULL; }
        if( accumbuf_free != NULL ) { opal_cuda_free_gpu_buffer(accumbuf_free, 0); accumbuf_free = NULL; }
    }

    /* leaf nodes
       Depending on the value of max_outstanding_reqs and
       the number of segments we have two options:
       - send all segments using blocking send to the parent, or
       - avoid overflooding the parent nodes by limiting the number of
       outstanding requests to max_oustanding_reqs.
       TODO/POSSIBLE IMPROVEMENT: If there is a way to determine the eager size
       for the current communication, synchronization should be used only
       when the message/segment size is smaller than the eager size.
    */
    else {

        /* If the number of segments is less than a maximum number of oustanding
           requests or there is no limit on the maximum number of outstanding
           requests, we send data to the parent using blocking send */
        if ((0 == max_outstanding_reqs) ||
            (num_segments <= max_outstanding_reqs)) {

            segindex = 0;
            while ( original_count > 0) {
                if (original_count < count_by_segment) {
                    count_by_segment = original_count;
                }
                ret = MCA_PML_CALL( send((char*)sendbuf +
                                         (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                         count_by_segment, datatype,
                                         tree->tree_prev,
                                         MCA_COLL_BASE_TAG_REDUCE,
                                         MCA_PML_BASE_SEND_STANDARD,
                                         comm) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
                segindex++;
                original_count -= count_by_segment;
            }
        }

        /* Otherwise, introduce flow control:
           - post max_outstanding_reqs non-blocking synchronous send,
           - for remaining segments
           - wait for a ssend to complete, and post the next one.
           - wait for all outstanding sends to complete.
        */
        else {

            int creq = 0;

            sreq = coll_base_comm_get_reqs(module->base_data, max_outstanding_reqs);
            if (NULL == sreq) { line = __LINE__; ret = -1; goto error_hndl; }

            /* post first group of requests */
            for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[segindex]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                original_count -= count_by_segment;
            }

            creq = 0;
            while ( original_count > 0 ) {
                /* wait on a posted request to complete */
                ret = ompi_request_wait(&sreq[creq], MPI_STATUS_IGNORE);
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }

                if( original_count < count_by_segment ) {
                    count_by_segment = original_count;
                }
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[creq]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                creq = (creq + 1) % max_outstanding_reqs;
                segindex++;
                original_count -= count_by_segment;
            }

            /* Wait on the remaining request to complete */
            ret = ompi_request_wait_all( max_outstanding_reqs, sreq,
                                         MPI_STATUSES_IGNORE );
            if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
        }
    }
    return OMPI_SUCCESS;

 error_hndl:  /* error handler */
    OPAL_OUTPUT (( ompi_coll_base_framework.framework_output,
                   "ERROR_HNDL: node %d file %s line %d error %d\n",
                   rank, __FILE__, line, ret ));
    (void)line;  // silence compiler warning
    if( inbuf_free[0] != NULL ) { opal_cuda_free_gpu_buffer(inbuf_free[0], 0); inbuf_free[0] = NULL; }
    if( inbuf_free[1] != NULL ) { opal_cuda_free_gpu_buffer(inbuf_free[1], 0); inbuf_free[1] = NULL; }
    if( accumbuf_free != NULL ) { opal_cuda_free_gpu_buffer(accumbuf, 0); accumbuf = NULL; }
    if( NULL != sreq ) {
        ompi_coll_base_free_reqs(sreq, max_outstanding_reqs);
    }
    return ret;
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

int mca_coll_adapt_cuda_reduce_topo_generic_cpu( const void* sendbuf, void* recvbuf, int original_count,
                                    struct ompi_datatype_t* datatype, struct ompi_op_t* op,
                                    int root, struct ompi_communicator_t* comm,
                                    mca_coll_base_module_t *module,
                                    ompi_coll_tree_t* tree, int count_by_segment,
                                    int max_outstanding_reqs )
{
    char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
    char *accumbuf = NULL, *accumbuf_free = NULL;
    char *local_op_buffer = NULL, *sendtmpbuf = NULL;
    ptrdiff_t extent, size, gap, segment_increment;
    ompi_request_t **sreq = NULL, *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int num_segments, line, ret, segindex, i, rank;
    int recvcount, prevcount, inbi;

    /**
     * Determine number of segments and number of elements
     * sent per operation
     */
    ompi_datatype_type_extent( datatype, &extent );
    num_segments = (int)(((size_t)original_count + (size_t)count_by_segment - (size_t)1) / (size_t)count_by_segment);
    segment_increment = (ptrdiff_t)count_by_segment * extent;

    sendtmpbuf = (char*) sendbuf;
    if( sendbuf == MPI_IN_PLACE ) {
        sendtmpbuf = (char *)recvbuf;
    }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:reduce_generic count %d, msg size %ld, segsize %ld, max_requests %d",
                 original_count, (unsigned long)((ptrdiff_t)num_segments * (ptrdiff_t)segment_increment),
                 (unsigned long)segment_increment, max_outstanding_reqs));

    rank = ompi_comm_rank(comm);
    
    char *cpu_buff[2] = {NULL, NULL}, *cpu_buff_free[2] = {NULL, NULL};
    char *gpu_buff[2] = {NULL, NULL};
    char *local_send_buff_tmp = NULL;
    char *local_recv_buff_tmp = NULL;
    size_t type_size;
    ompi_datatype_type_size(datatype, &type_size);
    mca_mpool_base_module_t *mpool = mca_coll_adapt_cuda_component.pined_cpu_mpool;
    ptrdiff_t real_segment_size;

    /* non-leaf nodes - wait for children to send me data & forward up
       (if needed) */
    print_topo_level(rank, tree);
    if( tree->tree_nextsize > 0 ) {

        /* handle non existant recv buffer (i.e. its NULL) and
           protect the recv buffer on non-root nodes */
        accumbuf = (char*)recvbuf;
        if( (NULL == accumbuf) || (root != rank) ) {
            /* Allocate temporary accumulator buffer. */
            size = opal_datatype_span(&datatype->super, original_count, &gap);
            accumbuf_free = (char*)opal_cuda_malloc_gpu_buffer(size, 0);
            if (accumbuf_free == NULL) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            accumbuf = accumbuf_free - gap;
        }

        /* If this is a non-commutative operation we must copy
           sendbuf to the accumbuf, in order to simplfy the loops */
        if (!ompi_op_is_commute(op)) {
            assert(0);
            ompi_datatype_copy_content_same_ddt(datatype, original_count,
                                                (char*)accumbuf,
                                                (char*)sendtmpbuf);
        }
        /* Allocate two buffers for incoming segments */
        real_segment_size = opal_datatype_span(&datatype->super, count_by_segment, &gap);
        inbuf_free[0] = (char*)opal_cuda_malloc_gpu_buffer(real_segment_size, 0);
        if( inbuf_free[0] == NULL ) {
            line = __LINE__; ret = -1; goto error_hndl;
        }
        inbuf[0] = inbuf_free[0] - gap;
        
        /* node and socket leader allocate cpu buffer */
        if (tree->topo_flags == 1 || tree->topo_flags == 0) {
            cpu_buff_free[0] = (char*) mpool->mpool_alloc(mpool, real_segment_size, 0, 0);
            if( cpu_buff_free[0] == NULL ) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            cpu_buff[0] = cpu_buff_free[0] - gap;
        }
        
        /* if there is chance to overlap communication -
           allocate second buffer */
        if( (num_segments > 1) || (tree->tree_nextsize > 1) ) {
            inbuf_free[1] = (char*)opal_cuda_malloc_gpu_buffer(real_segment_size, 0);
            if( inbuf_free[1] == NULL ) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            inbuf[1] = inbuf_free[1] - gap;
            
            /* node and socket leader allocate cpu buffer */
            if (tree->topo_flags == 1 || tree->topo_flags == 0) {
                cpu_buff_free[1] = (char*) mpool->mpool_alloc(mpool, real_segment_size, 0, 0);
                if( cpu_buff_free[1] == NULL ) {
                    line = __LINE__; ret = -1; goto error_hndl;
                }
                cpu_buff[1] = cpu_buff_free[1] - gap;
            }
        }

        /* reset input buffer index and receive count */
        inbi = 0;
        recvcount = 0;
        /* for each segment */
        for( segindex = 0; segindex <= num_segments; segindex++ ) {
            prevcount = recvcount;
            /* recvcount - number of elements in current segment */
            recvcount = count_by_segment;
            if( segindex == (num_segments-1) )
                recvcount = original_count - (ptrdiff_t)count_by_segment * (ptrdiff_t)segindex;

            /* for each child */
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                /**
                 * We try to overlap communication:
                 * either with next segment or with the next child
                 */
                /* post irecv for current segindex on current child */
                if( segindex < num_segments ) {
                    void* local_recvbuf = inbuf[inbi];
                    if( 0 == i ) {
                        /* for the first step (1st child per segment) and
                         * commutative operations we might be able to irecv
                         * directly into the accumulate buffer so that we can
                         * reduce(op) this with our sendbuf in one step as
                         * ompi_op_reduce only has two buffer pointers,
                         * this avoids an extra memory copy.
                         *
                         * BUT if the operation is non-commutative or
                         * we are root and are USING MPI_IN_PLACE this is wrong!
                         */
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_recvbuf = accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }
                    
                    /* node leader go through cpu */
                    if (tree->topo_flags == 0 && tree->tree_next_topo_flags[i] != 2) {
                      //  opal_cuda_memcpy_sync(cpu_buff[inbi], local_recvbuf, recvcount*type_size);
                        local_recv_buff_tmp = cpu_buff[inbi];
                        gpu_buff[inbi] = local_recvbuf;
                    } else {
                        local_recv_buff_tmp = local_recvbuf;
                        gpu_buff[inbi] = NULL;
                    }

                    ret = MCA_PML_CALL(irecv(local_recv_buff_tmp, recvcount, datatype,
                                             tree->tree_next[i],
                                             MCA_COLL_BASE_TAG_REDUCE, comm,
                                             &reqs[inbi]));
                    if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;}
                }
                /* wait for previous req to complete, if any.
                   if there are no requests reqs[inbi ^1] will be
                   MPI_REQUEST_NULL. */
                /* wait on data from last child for previous segment */
                ret = ompi_request_wait_all( 1, &reqs[inbi ^ 1],
                                             MPI_STATUSES_IGNORE );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                
                /* node leader  copy back to gpu */
                if (tree->topo_flags == 0) {
                    if (i > 0) {
                        if (cpu_buff[inbi^1] != NULL && gpu_buff[inbi^1] != NULL) {
                            opal_cuda_memcpy_sync(gpu_buff[inbi ^ 1], cpu_buff[inbi ^ 1], recvcount*type_size);
                            gpu_buff[inbi^1] == NULL;
                        }
                    } else if (segindex > 0) {
                        if (cpu_buff[inbi^1] != NULL && gpu_buff[inbi^1] != NULL) {
                            opal_cuda_memcpy_sync(gpu_buff[inbi ^ 1], cpu_buff[inbi ^ 1], prevcount*type_size);
                            gpu_buff[inbi^1] == NULL;
                        }
                    }
                }
                
                local_op_buffer = inbuf[inbi ^ 1];
                if( i > 0 ) {
                    /* our first operation is to combine our own [sendbuf] data
                     * with the data we recvd from down stream (but only
                     * the operation is commutative and if we are not root and
                     * not using MPI_IN_PLACE)
                     */
                    if( 1 == i ) {
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }
                    /* apply operation */
                    opal_cuda_recude_op_sum_double(local_op_buffer, accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment, recvcount, NULL);
                } else if ( segindex > 0 ) {
                    void* accumulator = accumbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                    if( tree->tree_nextsize <= 1 ) {
                        if( (ompi_op_is_commute(op)) &&
                            !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                        }
                    }
                    opal_cuda_recude_op_sum_double(local_op_buffer, accumulator, prevcount, NULL);

                    /* all reduced on available data this step (i) complete,
                     * pass to the next process unless you are the root.
                     */
                    if (rank != tree->tree_root) {
                        if (tree->topo_flags == 0 || tree->topo_flags == 1) {
                            opal_cuda_memcpy_sync(cpu_buff[inbi ^ 1], accumulator, prevcount*type_size);
                            local_send_buff_tmp = cpu_buff[inbi ^ 1];
                        } else {
                            local_send_buff_tmp = accumulator;
                        }
                        /* send combined/accumulated data to parent */
                        ret = MCA_PML_CALL( send( local_send_buff_tmp, prevcount,
                                                  datatype, tree->tree_prev,
                                                  MCA_COLL_BASE_TAG_REDUCE,
                                                  MCA_PML_BASE_SEND_STANDARD,
                                                  comm) );
                        if (ret != MPI_SUCCESS) {
                            line = __LINE__; goto error_hndl;
                        }
                    }

                    /* we stop when segindex = number of segments
                       (i.e. we do num_segment+1 steps for pipelining */
                    if (segindex == num_segments) break;
                }

                /* update input buffer index */
                inbi = inbi ^ 1;
            } /* end of for each child */
        } /* end of for each segment */

        /* clean up */
        if( inbuf_free[0] != NULL) { opal_cuda_free_gpu_buffer(inbuf_free[0], 0); inbuf_free[0] = NULL; }
        if( inbuf_free[1] != NULL) { opal_cuda_free_gpu_buffer(inbuf_free[1], 0); inbuf_free[1] = NULL; }
        if( accumbuf_free != NULL ) { opal_cuda_free_gpu_buffer(accumbuf_free, 0); accumbuf_free = NULL; }
    }

    /* leaf nodes
       Depending on the value of max_outstanding_reqs and
       the number of segments we have two options:
       - send all segments using blocking send to the parent, or
       - avoid overflooding the parent nodes by limiting the number of
       outstanding requests to max_oustanding_reqs.
       TODO/POSSIBLE IMPROVEMENT: If there is a way to determine the eager size
       for the current communication, synchronization should be used only
       when the message/segment size is smaller than the eager size.
    */
    else {

        /* If the number of segments is less than a maximum number of oustanding
           requests or there is no limit on the maximum number of outstanding
           requests, we send data to the parent using blocking send */
        if ((0 == max_outstanding_reqs) ||
            (num_segments <= max_outstanding_reqs)) {

            segindex = 0;
            while ( original_count > 0) {
                if (original_count < count_by_segment) {
                    count_by_segment = original_count;
                }
                /* socket leader */
                if ((tree->topo_flags == 1 || tree->topo_flags == 0) && tree->tree_prev_topo_flags == 0) {
                    if (cpu_buff_free[0] == NULL) {
                        real_segment_size = opal_datatype_span(&datatype->super, count_by_segment, &gap);
                        cpu_buff_free[0] = (char*) mpool->mpool_alloc(mpool, real_segment_size, 0, 0);
                        if( cpu_buff_free[0] == NULL ) {
                            line = __LINE__; ret = -1; goto error_hndl;
                        }
                        cpu_buff[0] = cpu_buff_free[0] - gap;
                    }
                    assert(cpu_buff[0] != NULL);
                    opal_cuda_memcpy_sync(cpu_buff[0], (char*)sendbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment, count_by_segment*type_size);
                    local_send_buff_tmp = cpu_buff[0];
                } else {
                    local_send_buff_tmp = (char*)sendbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                }
                ret = MCA_PML_CALL( send((char*)local_send_buff_tmp,
                                         count_by_segment, datatype,
                                         tree->tree_prev,
                                         MCA_COLL_BASE_TAG_REDUCE,
                                         MCA_PML_BASE_SEND_STANDARD,
                                         comm) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
                segindex++;
                original_count -= count_by_segment;
            }
        }

        /* Otherwise, introduce flow control:
           - post max_outstanding_reqs non-blocking synchronous send,
           - for remaining segments
           - wait for a ssend to complete, and post the next one.
           - wait for all outstanding sends to complete.
        */
        else {
            assert(0);

            int creq = 0;

            sreq = coll_base_comm_get_reqs(module->base_data, max_outstanding_reqs);
            if (NULL == sreq) { line = __LINE__; ret = -1; goto error_hndl; }

            /* post first group of requests */
            for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[segindex]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                original_count -= count_by_segment;
            }

            creq = 0;
            while ( original_count > 0 ) {
                /* wait on a posted request to complete */
                ret = ompi_request_wait(&sreq[creq], MPI_STATUS_IGNORE);
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }

                if( original_count < count_by_segment ) {
                    count_by_segment = original_count;
                }
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[creq]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                creq = (creq + 1) % max_outstanding_reqs;
                segindex++;
                original_count -= count_by_segment;
            }

            /* Wait on the remaining request to complete */
            ret = ompi_request_wait_all( max_outstanding_reqs, sreq,
                                         MPI_STATUSES_IGNORE );
            if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
        }
    }
    
    if (cpu_buff_free[0] != NULL) { mpool->mpool_free(mpool, cpu_buff_free[0]); cpu_buff_free[0] = NULL; }
    if (cpu_buff_free[1] != NULL) { mpool->mpool_free(mpool, cpu_buff_free[1]); cpu_buff_free[1] = NULL;}
    return OMPI_SUCCESS;

 error_hndl:  /* error handler */
    OPAL_OUTPUT (( ompi_coll_base_framework.framework_output,
                   "ERROR_HNDL: node %d file %s line %d error %d\n",
                   rank, __FILE__, line, ret ));
    (void)line;  // silence compiler warning
    if( inbuf_free[0] != NULL ) { opal_cuda_free_gpu_buffer(inbuf_free[0], 0); inbuf_free[0] = NULL; }
    if( inbuf_free[1] != NULL ) { opal_cuda_free_gpu_buffer(inbuf_free[1], 0); inbuf_free[1] = NULL; }
    if( accumbuf_free != NULL ) { opal_cuda_free_gpu_buffer(accumbuf, 0); accumbuf = NULL; }
    if( NULL != sreq ) {
        ompi_coll_base_free_reqs(sreq, max_outstanding_reqs);
    }
    return ret;
}


static mca_coll_adapt_cuda_inbuf_t * to_inbuf(char * buf, int distance){
    return (mca_coll_adapt_cuda_inbuf_t *)(buf - distance);
}

static int send_cb(ompi_request_t *req){
    mca_coll_adapt_cuda_reduce_context_t *context = (mca_coll_adapt_cuda_reduce_context_t *) req->req_complete_cb_data;
    TEST("[%d]: send_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id);
    int err;
    
    opal_atomic_sub_32(&(context->con->ongoing_send), 1);
    
    //send a new segment
    //list is not empty
    opal_mutex_lock (context->con->mutex_recv_list);
    mca_coll_adapt_cuda_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    if (item != NULL) {
        //get new context item from free list
        mca_coll_adapt_cuda_reduce_context_t * send_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        if (context->con->tree->tree_nextsize > 0) {
            send_context->buff = context->con->accumbuf[item->id];

        }
        else{
            send_context->buff = context->buff + (item->id - context->frag_id) * context->con->segment_increment;
        }
        send_context->frag_id = item->id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        
        opal_atomic_add_32(&(context->con->ongoing_send), 1);
        
        int send_count = send_context->con->seg_count;
        if (item->id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - item->id * send_context->con->seg_count;
        }
        
        TEST("[%d]: In send_cb, create isend to seg %d, peer %d\n", send_context->con->rank, send_context->frag_id, send_context->peer);
        
        ompi_request_t *send_req;
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        
        //release the item
        OBJ_RELEASE(item);
        
        //invoke send call back
        ompi_request_set_callback(send_req, send_cb, send_context);
    }
    
    opal_mutex_lock(context->con->mutex_num_sent);
    int32_t num_sent = ++(context->con->num_sent_segs);
    TEST("[%d]: In send_cb, root = %d, num_sent = %d, num_segs = %d\n", context->con->rank, context->con->tree->tree_root, num_sent, context->con->num_segs);
    //check whether signal the condition, non root and sent all the segments
    if (context->con->tree->tree_root != context->con->rank && num_sent == context->con->num_segs) {
        opal_mutex_unlock(context->con->mutex_num_sent);
        TEST("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
        int i;
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        if (context->con->accumbuf != NULL) {
            if (context->con->rank != context->con->root ) {
                for (i=0; i<context->con->num_segs; i++) {
                    if (context->con->accumbuf[i] != NULL) {
                        opal_cuda_free_gpu_buffer(context->con->accumbuf[i], 0);
                    }
                    //opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)to_inbuf(context->con->accumbuf[i], context->con->distance));
                }
            }
            free(context->con->accumbuf);
        }
        OBJ_RELEASE(context->con->recv_list);
        for (i=0; i<context->con->num_segs; i++) {
            OBJ_RELEASE(context->con->mutex_op_list[i]);
        }
        free(context->con->mutex_op_list);
        OBJ_RELEASE(context->con->mutex_num_recv_segs);
        OBJ_RELEASE(context->con->mutex_recv_list);
        OBJ_RELEASE(context->con->mutex_num_sent);
        if (context->con->tree->tree_nextsize > 0) {
         //   OBJ_RELEASE(context->con->inbuf_list);
            free(context->con->next_recv_segs);
        }
        OBJ_RELEASE(context->con->context_list);
        OBJ_RELEASE(context->con);
        ompi_request_complete(temp_req, 1);
    }
    else{
        opal_mutex_unlock(context->con->mutex_num_sent);
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
   // no lock OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_cuda_reduce_context_t *context = (mca_coll_adapt_cuda_reduce_context_t *) req->req_complete_cb_data;
    TEST("[%d]: recv_cb, peer %d, seg_id %d\n", context->con->rank, context->peer, context->frag_id);
    
    int err;
    //atomic
    int32_t new_id = opal_atomic_add_32(&(context->con->next_recv_segs[context->child_id]), 1);
    
    //receive new segment
    if (new_id < context->con->num_segs) {
        char * temp_recv_buf = NULL;
        char * inbuf = NULL;
        //set inbuf, if it it first child, recv on rbuf, else recv on inbuf
        if (context->child_id == 0 && context->con->sbuf != MPI_IN_PLACE && context->con->root == context->con->rank) {
            temp_recv_buf = (char *)context->con->rbuf + (ptrdiff_t)new_id * (ptrdiff_t)context->con->segment_increment;
        }
        else {
            inbuf = opal_cuda_malloc_gpu_buffer(context->con->real_seg_size, 0);
            temp_recv_buf = inbuf - context->con->lower_bound;
        }
        //get new context item from free list
        mca_coll_adapt_cuda_reduce_context_t * recv_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        recv_context->buff = temp_recv_buf;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        recv_context->inbuf = inbuf;
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        TEST("[%d]: In recv_cb, create irecv for seg %d, peer %d, inbuf %p\n", context->con->rank, recv_context->frag_id, recv_context->peer, (void *)inbuf);
        ompi_request_t *recv_req;
        MCA_PML_CALL(irecv(temp_recv_buf, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        ompi_request_set_callback(recv_req, recv_cb, recv_context);
    }
    
    //do the op
    int op_count = context->con->seg_count;
    if (context->frag_id == (context->con->num_segs - 1)) {
        op_count = context->con->count - context->frag_id * context->con->seg_count;
    }
    
    int keep_inbuf = 0;
    opal_mutex_lock(context->con->mutex_op_list[context->frag_id]);
    if (context->con->accumbuf[context->frag_id] == NULL) {
        if (context->inbuf == NULL) {
            TEST("[%d]: set accumbuf to rbuf\n", context->con->rank);
            context->con->accumbuf[context->frag_id] = context->buff;
        }
        else {
            keep_inbuf = 1;
            TEST("[%d]: set accumbuf to inbuf\n", context->con->rank);
            context->con->accumbuf[context->frag_id] = context->inbuf - context->con->lower_bound;
        }
        //op sbuf and accmbuf to accumbuf
       // ompi_op_reduce(context->con->op, context->con->sbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment, context->con->accumbuf[context->frag_id], op_count, context->con->datatype);
        opal_cuda_recude_op_sum_double(context->con->sbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment, context->con->accumbuf[context->frag_id], op_count, NULL);

    }
    else {
        if (context->inbuf == NULL) {
            //op rbuf and accumbuf to rbuf
            TEST("[%d]: op rbuf and accumbuf to rbuf\n", context->con->rank);
            //ompi_op_reduce(context->con->op, context->con->accumbuf[context->frag_id], context->buff, op_count, context->con->datatype);
            opal_cuda_recude_op_sum_double(context->con->accumbuf[context->frag_id], context->buff, op_count, NULL);
            //free old accumbuf
            opal_cuda_free_gpu_buffer(context->con->accumbuf[context->frag_id], 0);
            //opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)to_inbuf(context->con->accumbuf[context->frag_id], context->con->distance));
            //set accumbut to rbuf
            context->con->accumbuf[context->frag_id] = context->buff;
        }
        else {
            //op inbuf and accmbuf to accumbuf
            TEST("[%d]: op inbuf and accmbuf to accumbuf\n", context->con->rank);
            //ompi_op_reduce(context->con->op, context->inbuf->buff - context->con->lower_bound, context->con->accumbuf[context->frag_id], op_count, context->con->datatype);
            opal_cuda_recude_op_sum_double(context->inbuf - context->con->lower_bound, context->con->accumbuf[context->frag_id], op_count, NULL);
        }
    }

    opal_mutex_unlock(context->con->mutex_op_list[context->frag_id]);
    
    //set recv list
    opal_mutex_lock (context->con->mutex_recv_list);
    add_to_list(context->con->recv_list, context->frag_id);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    //send to parent
    if (context->con->rank != context->con->tree->tree_root && context->con->ongoing_send < SEND_NUM) {
        //atomic
        opal_mutex_lock (context->con->mutex_recv_list);
        mca_coll_adapt_cuda_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
        opal_mutex_unlock (context->con->mutex_recv_list);
        
        if (item != NULL) {
            //get new context item from free list
            mca_coll_adapt_cuda_reduce_context_t * send_context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->con->accumbuf[context->frag_id];
            send_context->frag_id = item->id;
            send_context->peer = context->con->tree->tree_prev;
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            //atomic
            opal_atomic_add_32(&(context->con->ongoing_send), 1);
            
            int send_count = send_context->con->seg_count;
            if (item->id == (send_context->con->num_segs - 1)) {
                send_count = send_context->con->count - item->id * send_context->con->seg_count;
            }
            
            TEST("[%d]: In recv_cb, create isend to seg %d, peer %d\n", send_context->con->rank, send_context->frag_id, send_context->peer);
            
            ompi_request_t *send_req;
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //release the item
            OBJ_RELEASE(item);
            
            //invoke send call back
            ompi_request_set_callback(send_req, send_cb, send_context);
        }
    }
    
    opal_mutex_lock (context->con->mutex_num_recv_segs);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    TEST("[%d]: In recv_cb, root = %d, num_recv = %d, num_segs = %d, num_child = %d\n", context->con->rank, context->con->tree->tree_root, num_recv_segs_t, context->con->num_segs, context->con->tree->tree_nextsize);
    //if this is root and has received all the segments
    if (context->con->tree->tree_root == context->con->rank && num_recv_segs_t == context->con->num_segs * context->con->tree->tree_nextsize) {
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        int i;
        TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
        ompi_request_t *temp_req = context->con->request;
        if (!keep_inbuf && context->inbuf != NULL) {
            opal_cuda_free_gpu_buffer(context->inbuf, 0);
           // opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        if (context->con->accumbuf != NULL) {
            if (context->con->rank != context->con->root) {
                for (i=0; i<context->con->num_segs; i++) {
                    if (context->con->accumbuf[i]!= NULL) {
                        opal_cuda_free_gpu_buffer(context->con->accumbuf[i], 0);
                    }
                //    opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)to_inbuf(context->con->accumbuf[i], context->con->distance));
                }
            }
            free(context->con->accumbuf);
        }
        OBJ_RELEASE(context->con->recv_list);
        for (i=0; i<context->con->num_segs; i++) {
            OBJ_RELEASE(context->con->mutex_op_list[i]);
        }
        free(context->con->mutex_op_list);
        OBJ_RELEASE(context->con->mutex_num_recv_segs);
        OBJ_RELEASE(context->con->mutex_recv_list);
        OBJ_RELEASE(context->con->mutex_num_sent);
        if (context->con->tree->tree_nextsize > 0) {
         //   OBJ_RELEASE(context->con->inbuf_list);
            free(context->con->next_recv_segs);
        }
        OBJ_RELEASE(context->con->context_list);
        OBJ_RELEASE(context->con);
        ompi_request_complete(temp_req, 1);
    }
    else{
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        if (!keep_inbuf && context->inbuf != NULL) {
            TEST("return inbuf\n");
            opal_cuda_free_gpu_buffer(context->inbuf, 0);
           // opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        TEST("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    // no lock OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_cuda_reduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    
    ptrdiff_t extent, lower_bound, segment_increment;
    ptrdiff_t true_lower_bound, true_extent, real_seg_size;
    size_t typelng;
    int seg_count = count, num_segs, rank, recv_count, send_count, i, j, err, min, distance = 0;
    int32_t seg_index;
    int * next_recv_segs = NULL;
    char **accumbuf = NULL;      //used to store the accumuate result, pointer to every segment
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    opal_free_list_t * inbuf_list; //a free list contain all recv data
    opal_mutex_t * mutex_recv_list;
    opal_mutex_t * mutex_num_recv_segs;
    opal_mutex_t * mutex_num_sent;
    opal_mutex_t ** mutex_op_list;
    opal_list_t * recv_list;     //a list to store the segments need to be sent
    
    // Determine number of segments and number of elements sent per operation
    rank = ompi_comm_rank(comm);
    ompi_datatype_get_extent( dtype, &lower_bound, &extent );
    ompi_datatype_type_size( dtype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( SEG_SIZE, typelng, seg_count );
    num_segs = (count + seg_count - 1) / seg_count;
    segment_increment = (ptrdiff_t)seg_count * extent;
    ompi_datatype_get_true_extent(dtype, &true_lower_bound, &true_extent);
    real_seg_size = true_extent + (ptrdiff_t)(seg_count - 1) * extent;
    
    if (rank == root) printf("reduce cuda generic\n");
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_cuda_reduce_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_cuda_reduce_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //not leaf
    
    if (tree->tree_nextsize > 0) {
        // inbuf_list = OBJ_NEW(opal_free_list_t);
        // opal_free_list_init(inbuf_list,
        //                     sizeof(mca_coll_adapt_cuda_inbuf_t),
        //                     opal_cache_line_size,
        //                     OBJ_CLASS(mca_coll_adapt_cuda_inbuf_t),
        //                     0,opal_cache_line_size,
        //                     FREE_LIST_NUM_INBUF_LIST,
        //                     FREE_LIST_MAX_INBUF_LIST,
        //                     FREE_LIST_INC_INBUF_LIST,
        //                     NULL, 0, NULL, NULL, NULL);
        //set up next_recv_segs
        next_recv_segs = (int32_t *)malloc(sizeof(int32_t) * tree->tree_nextsize);
        inbuf_list = NULL;
        // mca_coll_adapt_cuda_inbuf_t * temp_inbuf = (mca_coll_adapt_cuda_inbuf_t *) opal_free_list_wait(inbuf_list);
        // distance = (char *)temp_inbuf->buff - lower_bound - (char *)temp_inbuf; //address of inbuf->buff to address of inbuf
        // opal_free_list_return(inbuf_list, (opal_free_list_item_t*)temp_inbuf);
    }
    else {
        inbuf_list = NULL;
        next_recv_segs = NULL;
    }
    
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
    
    //set up mutex
    mutex_recv_list = OBJ_NEW(opal_mutex_t);
    mutex_num_recv_segs = OBJ_NEW(opal_mutex_t);
    mutex_op_list = (opal_mutex_t **)malloc(sizeof(opal_mutex_t *) * num_segs);
    for (i=0; i<num_segs; i++) {
        mutex_op_list[i] = OBJ_NEW(opal_mutex_t);
    }
    mutex_num_sent = OBJ_NEW(opal_mutex_t);
    //create recv_list
    recv_list = OBJ_NEW(opal_list_t);
    
    //Set constant context for send and recv call back
    mca_coll_adapt_cuda_constant_reduce_context_t *con = OBJ_NEW(mca_coll_adapt_cuda_constant_reduce_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = dtype;
    con->comm = comm;
    con->segment_increment = segment_increment;
    con->num_segs = num_segs;
    con->request = temp_request;
    con->rank = rank;
    con->context_list = context_list;
    con->num_recv_segs = 0;
    con->num_sent_segs = 0;
    con->next_recv_segs = next_recv_segs;
    con->mutex_recv_list = mutex_recv_list;
    con->mutex_num_recv_segs = mutex_num_recv_segs;
    con->mutex_num_sent = mutex_num_sent;
    con->mutex_op_list = mutex_op_list;
    con->op = op;
    con->tree = tree;
    con->inbuf_list = inbuf_list;
    con->recv_list = recv_list;
    con->lower_bound = lower_bound;
    con->ongoing_send = 0;
    con->sbuf = (char *)sbuf;
    con->rbuf = (char *)rbuf;
    con->root = root;
    con->distance = distance;
    con->real_seg_size = real_seg_size;
    // non leaf nodes
    if (tree->tree_nextsize > 0) {
        //set accumbuf
        accumbuf = (char **) malloc (sizeof(char*) * num_segs);
        if (root == rank && sbuf == MPI_IN_PLACE) {
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = (char *)rbuf + (ptrdiff_t)i * (ptrdiff_t)segment_increment;
            }
        }
        else{
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = NULL;
            }
        }
        
        con->accumbuf = accumbuf;
        
        //for the first batch of segments
        if (num_segs <= RECV_NUM) {
            min = num_segs;
        }
        else{
            min = RECV_NUM;
        }
        for (i=0; i<tree->tree_nextsize; i++) {
            next_recv_segs[i] = min - 1;
        }
        
        for( j = 0; j < min; j++ ) {
            //for each child
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                seg_index = j;
                if (seg_index < num_segs) {
                    recv_count = seg_count;
                    if( seg_index == (num_segs-1) ){
                        recv_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)seg_index;
                    }
                    char * temp_recv_buf = NULL;
                    char * inbuf = NULL;
                    //set inbuf, if it it first child, recv on rbuf, else recv on inbuf
                    if (i==0 && sbuf != MPI_IN_PLACE && root == rank) {
                        temp_recv_buf = (char *)rbuf + (ptrdiff_t)j * (ptrdiff_t)segment_increment;
                    }
                    else {
                       // inbuf = (mca_coll_adapt_cuda_inbuf_t *) opal_free_list_wait(inbuf_list);
                        inbuf = opal_cuda_malloc_gpu_buffer(real_seg_size, 0);
                        temp_recv_buf = inbuf - lower_bound;
                    }
                    //get context
                    mca_coll_adapt_cuda_reduce_context_t * context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context_list);
                    context->buff = temp_recv_buf;
                    context->frag_id = seg_index;
                    context->child_id = i;              //the id of peer in in the tree
                    context->peer = tree->tree_next[i];   //the actural rank of the peer
                    context->con = con;
                    OBJ_RETAIN(con);
                    context->inbuf = inbuf;
                    
                    TEST("[%d]: In reduce, create irecv for seg %d, peer %d, recv_count %d, inbuf %p\n", context->con->rank, context->frag_id, context->peer, recv_count, (void *)inbuf);
                    
                    //create a recv request
                    ompi_request_t *recv_req;
                    err = MCA_PML_CALL(irecv(temp_recv_buf, recv_count, dtype, tree->tree_next[i], seg_index, comm, &recv_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke recv call back
                    ompi_request_set_callback(recv_req, recv_cb, context);
                }
            }
        }
        
    }
    
    //leaf nodes
    else{
        mca_coll_adapt_cuda_item_t *item;
        //set up recv_list
        for(seg_index = 0; seg_index < num_segs; seg_index++) {
            item = OBJ_NEW(mca_coll_adapt_cuda_item_t);
            item->id = seg_index;
            item->count = tree->tree_nextsize;
            opal_list_append(recv_list, (opal_list_item_t *)item);
        }
        if (num_segs <= SEND_NUM) {
            min = num_segs;
        }
        else{
            min = SEND_NUM;
        }
        con->accumbuf = accumbuf;
        for(i = 0; i < min; i++) {
            opal_mutex_lock (mutex_recv_list);
            item = get_next_ready_item(recv_list, tree->tree_nextsize);
            opal_mutex_unlock (mutex_recv_list);
            if (item != NULL) {
                send_count = seg_count;
                if(item->id == (num_segs-1)){
                    send_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)item->id;
                }
                mca_coll_adapt_cuda_reduce_context_t * context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context_list);
                context->buff = (char*)sbuf + (ptrdiff_t)item->id * (ptrdiff_t)segment_increment;
                context->frag_id = item->id;
                context->peer = tree->tree_prev;   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                context->inbuf = NULL;
                
                //atomic
                opal_atomic_add_32(&(context->con->ongoing_send), 1);
                TEST("[%d]: In reduce, create isend to seg %d, peer %d, send_count %d\n", context->con->rank, context->frag_id, context->peer, send_count);
                
                //create send request
                ompi_request_t *send_req;
                err = MCA_PML_CALL( isend(context->buff, send_count, dtype, tree->tree_prev, context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req) );
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //release the item
                OBJ_RELEASE(item);
                
                //invoke send call back
                ompi_request_set_callback(send_req, send_cb, context);
            }
        }
        
    }
    
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    
    return MPI_SUCCESS;
}

#if 0
int mca_coll_adapt_cuda_reduce_chain_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    
    ptrdiff_t extent, lower_bound, segment_increment;
    ptrdiff_t true_lower_bound, true_extent, real_seg_size;
    size_t typelng;
    int seg_count = count, num_segs, rank, recv_count, send_count, i, j, err, min;
    int32_t seg_index;
    int * next_recv_segs = NULL;
    char **accumbuf = NULL;      //used to store the accumuate result
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    opal_mutex_t * mutex_recv_list;
    opal_mutex_t * mutex_num_recv_segs;
    opal_mutex_t ** mutex_op_list;
    opal_list_t * recv_list;     //a list to store the segments need to be sent
    
    // Determine number of segments and number of elements sent per operation
    rank = ompi_comm_rank(comm);
    ompi_datatype_get_extent( dtype, &lower_bound, &extent );
    ompi_datatype_type_size( dtype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( SEG_SIZE, typelng, seg_count );
    num_segs = (count + seg_count - 1) / seg_count;
    segment_increment = (ptrdiff_t)seg_count * extent;
    ompi_datatype_get_true_extent(dtype, &true_lower_bound, &true_extent);
    real_seg_size = true_extent + (ptrdiff_t)(seg_count - 1) * extent;
    real_seg_size_cuda = real_seg_size;
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_cuda_reduce_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_cuda_reduce_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //not leaf
    if (tree->tree_nextsize > 0) {
        //set up next_recv_segs
        next_recv_segs = (int32_t *)malloc(sizeof(int32_t) * tree->tree_nextsize);
    }
    else {
        next_recv_segs = NULL;
    }
    
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
    
    //set up mutex
    mutex_recv_list = OBJ_NEW(opal_mutex_t);
    mutex_num_recv_segs = OBJ_NEW(opal_mutex_t);
    mutex_op_list = (opal_mutex_t **)malloc(sizeof(opal_mutex_t *) * num_segs);
    for (i=0; i<num_segs; i++) {
        mutex_op_list[i] = OBJ_NEW(opal_mutex_t);
    }
    
    //create recv_list
    recv_list = OBJ_NEW(opal_list_t);
    
    
    //Set constant context for send and recv call back
    mca_coll_adapt_cuda_constant_reduce_context_t *con = OBJ_NEW(mca_coll_adapt_cuda_constant_reduce_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = dtype;
    con->comm = comm;
    con->segment_increment = segment_increment;
    con->num_segs = num_segs;
    con->request = temp_request;
    con->rank = rank;
    con->context_list = context_list;
    con->num_recv_segs = 0;
    con->num_sent_segs = 0;
    con->next_recv_segs = next_recv_segs;
    con->mutex_recv_list = mutex_recv_list;
    con->mutex_num_recv_segs = mutex_num_recv_segs;
    con->mutex_op_list = mutex_op_list;
    con->op = op;
    con->tree = tree;
    con->inbuf_list = NULL;
    con->recv_list = recv_list;
    con->lower_bound = lower_bound;
    con->ongoing_send = 0;
    con->sbuf = (char *)sbuf;
    
    // non leaf nodes
    if (tree->tree_nextsize > 0) {
        assert(tree->tree_nextsize == 1);
        
        //set accumbuf
        accumbuf = (char **) malloc (sizeof(char*) * num_segs);
        if (root == rank){
            // if (sbuf != MPI_IN_PLACE) {
            //     TIMER_DATA_TYPE tstart, tend;
            //     long total_time;
            //     GET_TIME(tstart);
            //     ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
            //     GET_TIME( tend );
            //     total_time = ELAPSED_TIME( tstart, tend );
            //     printf("memcpy %ld us", total_time);
            //
            // }
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = (char *)rbuf + (ptrdiff_t)i * (ptrdiff_t)segment_increment;
            }
        }
        else{
            for (i=0; i<num_segs; i++) {
                accumbuf[i] = NULL;
            }
        }
        
        con->accumbuf = accumbuf;
        
        //for the first batch of segments
        if (num_segs <= RECV_NUM) {
            min = num_segs;
        }
        else{
            min = RECV_NUM;
        }
        next_recv_segs[0] = min - 1;
        
        for( j = 0; j < min; j++ ) {
            //for each child
            seg_index = j;
            if (seg_index < num_segs) {
                recv_count = seg_count;
                if( seg_index == (num_segs-1) ){
                    recv_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)seg_index;
                }
                //get inbuf
                //wei mca_coll_adapt_cuda_inbuf_t * inbuf = (mca_coll_adapt_cuda_inbuf_t *) opal_free_list_wait(inbuf_list);
                char *inbuf = NULL;
                if (rank == root) {
                    inbuf = accumbuf[seg_index];
                } else {
                    inbuf = (char*)opal_cuda_malloc_gpu_buffer(real_seg_size_cuda, 0);
                }
                //get context
                mca_coll_adapt_cuda_reduce_context_t * context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context_list);
                context->buff = NULL;
                context->frag_id = seg_index;
                context->child_id = 0;              //the id of peer in in the tree
                context->peer = tree->tree_next[0];   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                context->inbuf = (mca_coll_adapt_cuda_inbuf_t *)inbuf;
                
                TEST("[%d]: In reduce, create irecv for seg %d, peer %d, recv_count %d\n", context->con->rank, context->frag_id, context->peer, recv_count);
                
                //create a recv request
                ompi_request_t *recv_req;
                err = MCA_PML_CALL(irecv(inbuf - lower_bound, recv_count, dtype, tree->tree_next[0], seg_index, comm, &recv_req));
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //invoke recv call back
                if(!ompi_request_set_callback(recv_req, recv_cb, context)) {
                    recv_cb(recv_req);
                }
            }
        }
        
    }
    
    //leaf nodes
    else{
        mca_coll_adapt_cuda_item_t *item;
        //set up recv_list
        for(seg_index = 0; seg_index < num_segs; seg_index++) {
            item = OBJ_NEW(mca_coll_adapt_cuda_item_t);
            item->id = seg_index;
            item->count = tree->tree_nextsize;
            opal_list_append(recv_list, (opal_list_item_t *)item);
        }
        if (num_segs <= SEND_NUM) {
            min = num_segs;
        }
        else{
            min = SEND_NUM;
        }
        for(i = 0; i < min; i++) {
            opal_mutex_lock (mutex_recv_list);
            item = get_next_ready_item(recv_list, tree->tree_nextsize);
            opal_mutex_unlock (mutex_recv_list);
            if (item != NULL) {
                send_count = seg_count;
                if(item->id == (num_segs-1)){
                    send_count = count - (ptrdiff_t)seg_count * (ptrdiff_t)item->id;
                }
                mca_coll_adapt_cuda_reduce_context_t * context = (mca_coll_adapt_cuda_reduce_context_t *) opal_free_list_wait(context_list);
                context->buff = (char*)sbuf + (ptrdiff_t)item->id * (ptrdiff_t)segment_increment;
                context->frag_id = item->id;
                context->peer = tree->tree_prev;   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                //atomic
                opal_atomic_add_32(&(context->con->ongoing_send), 1);
                TEST("[%d]: In reduce, create isend to seg %d, peer %d, send_count %d\n", context->con->rank, context->frag_id, context->peer);
                
                //create send request
                ompi_request_t *send_req;
                err = MCA_PML_CALL( isend(context->buff, send_count, dtype,
                                          tree->tree_prev,
                                          context->frag_id,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &send_req) );
                
                if (MPI_SUCCESS != err) {
                    return err;
                }
                
                //release the item
                OBJ_RELEASE(item);
                
                //invoke send call back
                if(!ompi_request_set_callback(send_req, send_cb, context)) {
                    send_cb(send_req);
                }
            }
        }
        
    }
    
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    
    if (accumbuf != NULL) {
        if (rank != root) {
            for (i=0; i<num_segs; i++) {
                //wei opal_free_list_return(inbuf_list, (opal_free_list_item_t*)to_inbuf(accumbuf[i], distance));
                if (accumbuf[i] != NULL) {
                    opal_cuda_free_gpu_buffer(accumbuf[i], 0);
                }
            }
        }
        free(accumbuf);
    }
    OBJ_RELEASE(con);
    OBJ_RELEASE(recv_list);
    for (i=0; i<num_segs; i++) {
        OBJ_RELEASE(mutex_op_list[i]);
    }
    free(mutex_op_list);
    OBJ_RELEASE(mutex_num_recv_segs);
    OBJ_RELEASE(mutex_recv_list);
    if (tree->tree_nextsize > 0) {
        free(next_recv_segs);
    }
    OBJ_RELEASE(context_list);
    return MPI_SUCCESS;
}
#endif
