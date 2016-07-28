#include "ompi_config.h"
#include "ompi/communicator/communicator.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "coll_adapt_item.h"
#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "ompi/mca/coll/base/coll_base_topo.h"  //build tree

#define SEND_NUM 2    //send how many fragments at once
#define RECV_NUM 3    //receive how many fragments at once
#define SEG_SIZE 163740   //size of a segment
#define FREE_LIST_NUM_CONTEXT_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_CONTEXT_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_CONTEXT_LIST 10    //The incresment of the context free list
#define FREE_LIST_NUM_INBUF_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_INBUF_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_INBUF_LIST 10    //The incresment of the context free list

//static int times = 0;   //how many times to invoke the reduce


static mca_coll_adapt_item_t * get_next_ready_item(opal_list_t* list, int num_children){
    mca_coll_adapt_item_t *item;
    if (opal_list_is_empty(list)) {
        return NULL;
    }
    for(item = (mca_coll_adapt_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->count == num_children) {
            opal_list_remove_item(list, (opal_list_item_t *)item);
            return item;
        }
    }
    return NULL;
}

static int add_to_list(opal_list_t* list, int id){
    mca_coll_adapt_item_t *item;
    int ret = 0;
    for(item = (mca_coll_adapt_item_t *) opal_list_get_first(list);
        item != (mca_coll_adapt_item_t *) opal_list_get_end(list);
        item = (mca_coll_adapt_item_t *) ((opal_list_item_t *)item)->opal_list_next) {
        if (item->id == id) {
            (item->count)++;
            ret = 1;
            break;
        }
    }
    if (ret == 0) {
        item = OBJ_NEW(mca_coll_adapt_item_t);
        item->id = id;
        item->count = 1;
        opal_list_append(list, (opal_list_item_t *)item);
        ret = 2;
    }
    //printf("add_to_list_return %d\n", ret);
    return ret;
}

static int send_cb(ompi_request_t *req){
    mca_coll_adapt_reduce_context_t *context = (mca_coll_adapt_reduce_context_t *) req->req_complete_cb_data;
    //printf("[%d]: send_cb\n", context->con->rank);
    int err;
    
    int32_t num_sent = opal_atomic_add_32(&(context->con->num_sent_segs), 1);
    opal_atomic_sub_32(&(context->con->ongoing_send), 1);
    
    //send a new segment
    //list is not empty
    opal_mutex_lock (context->con->mutex_recv_list);
    mca_coll_adapt_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    if (item != NULL) {
        //get new context item from free list
        //printf("wait context_list\n");
        mca_coll_adapt_reduce_context_t * send_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        send_context->buff = context->buff + (item->id - context->frag_id) * context->con->segment_increment;
        send_context->frag_id = item->id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        
        opal_atomic_add_32(&(context->con->ongoing_send), 1);
        
        int send_count = send_context->con->seg_count;
        if (item->id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - item->id * send_context->con->seg_count;
        }
        
        //int* ta1 = (int *)(context->buff + (item->id - context->frag_id) * context->con->segment_increment);
        //printf("[%d]: In send_cb, create isend to seg %d, peer %d, sbuf contains %d, %d\n", send_context->con->rank, send_context->frag_id, send_context->peer, ta1[0], ta1[1]);
        
        ompi_request_t *send_req;
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        
        //release the item
        OBJ_RELEASE(item);
        
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
            send_cb(send_req);
        }
        
    }
    
    //printf("[%d]: In send_cb, root = %d, num_sent = %d, num_segs = %d\n", context->con->rank, context->con->tree->tree_root, num_sent, context->con->num_segs);
    //check whether signal the condition, non root and sent all the segments
    if (context->con->tree->tree_root != context->con->rank && num_sent == context->con->num_segs) {
        //printf("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
        ompi_request_t *temp_req = context->con->request;
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        //printf("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        ompi_request_complete(temp_req, 1);
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        //printf("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    return MPI_SUCCESS;
}

static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_reduce_context_t *context = (mca_coll_adapt_reduce_context_t *) req->req_complete_cb_data;
    //printf("[%d]: recv_cb\n", context->con->rank);
    
    //int* ta1 = (int *)(context->con->accumbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment);
    //printf("[%d]: Before op, seg %d, abuf contains %d, %d\n", context->con->rank, context->frag_id, ta1[0], ta1[1]);
    
    //int* ta2 = (int *)(context->inbuf->buff - context->con->lower_bound);
    //printf("[%d]: Before op, seg %d, rbuf contains %d, %d\n", context->con->rank, context->frag_id, ta2[0], ta2[1]);
    
    int err;
    //atomic
    int32_t new_id = opal_atomic_add_32(&(context->con->next_recv_segs[context->child_id]), 1);
    
    //receive new segment
    if (new_id < context->con->num_segs) {
        //get inbuf
        //printf("wait inbuf_list\n");
        mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
        //get new context item from free list
        //printf("wait context_list\n");
        mca_coll_adapt_reduce_context_t * recv_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(context->con->context_list);
        //        recv_context->buff = context->buff + (new_id - context->frag_id) * context->con->segment_increment;
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
        
        //printf("[%d]: In recv_cb, create irecv for seg %d, peer %d\n", context->con->rank, recv_context->frag_id, recv_context->peer);
        
        ompi_request_t *recv_req;
        MCA_PML_CALL(irecv(recv_context->inbuf->buff - recv_context->con->lower_bound, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
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
    
    opal_mutex_lock(context->con->mutex_op_list[context->frag_id]);
    ompi_op_reduce(context->con->op, context->inbuf->buff - context->con->lower_bound,
                   context->con->accumbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment,
                   op_count, context->con->datatype);
    opal_mutex_unlock(context->con->mutex_op_list[context->frag_id]);
    
    //int* ta0 = (int *)(context->con->accumbuf + (ptrdiff_t)context->frag_id * (ptrdiff_t)context->con->segment_increment);
    //printf("[%d]: After op, seg %d, contains %d, %d\n", context->con->rank, context->frag_id, ta0[0], ta0[1]);
    
    //set recv list
    opal_mutex_lock (context->con->mutex_recv_list);
    add_to_list(context->con->recv_list, context->frag_id);
    opal_mutex_unlock (context->con->mutex_recv_list);
    
    //send to parent
    if (context->con->rank != context->con->tree->tree_root && context->con->ongoing_send < SEND_NUM) {
        //atomic
        opal_mutex_lock (context->con->mutex_recv_list);
        mca_coll_adapt_item_t *item = get_next_ready_item(context->con->recv_list, context->con->tree->tree_nextsize);
        opal_mutex_unlock (context->con->mutex_recv_list);
        
        if (item != NULL) {
            //get new context item from free list
            //printf("wait context_list\n");
            mca_coll_adapt_reduce_context_t * send_context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->con->accumbuf + item->id * context->con->segment_increment;
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
            
            //int* ta3 = (int *)(context->con->accumbuf + item->id * context->con->segment_increment);
            //printf("[%d]: In recv_cb, create isend to seg %d, peer %d, sbuf contains %d, %d\n", send_context->con->rank, send_context->frag_id, send_context->peer, ta3[0], ta3[1]);
            
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
    //printf("[%d]: In recv_cb, root = %d, num_recv = %d, num_segs = %d, num_child = %d\n", context->con->rank, context->con->tree->tree_root, num_recv_segs_t, context->con->num_segs, context->con->tree->tree_nextsize);
    //if this is root and has received all the segments
    if (context->con->tree->tree_root == context->con->rank && num_recv_segs_t == context->con->num_segs * context->con->tree->tree_nextsize) {
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        //printf("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
        ompi_request_t *temp_req = context->con->request;
        //printf("return inbuf\n");
        opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        //printf("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        ompi_request_complete(temp_req, 1);
    }
    else{
        opal_mutex_unlock (context->con->mutex_num_recv_segs);
        //printf("return inbuf\n");
        opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        //printf("return context_list\n");
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        
    }
    
    return MPI_SUCCESS;
}

int mca_coll_adapt_reduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    return mca_coll_adapt_reduce_topoaware_chain(sbuf, rbuf, count, dtype, op, root, comm, module);
}

int mca_coll_adapt_reduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_bmtree(comm, root);
    int r = mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_tree(2, comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}


int mca_coll_adapt_reduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(1, comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(4, comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    //TODO: has problem when comm_size = 2
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_topoaware_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_topoaware_linear(comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}

int mca_coll_adapt_reduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_topoaware_chain(comm, root);
    int r =  mca_coll_adapt_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return r;
}



int mca_coll_adapt_reduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    
    ptrdiff_t extent, lower_bound, segment_increment;
    ptrdiff_t true_lower_bound, true_extent, real_seg_size;
    size_t typelng;
    int seg_count = count, num_segs, rank, recv_count, send_count, i, j, err, min;
    int32_t seg_index;
    int * next_recv_segs = NULL;
    char *accumbuf = NULL, *accumbuf_free = NULL;      //used to store the accumuate result
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    opal_free_list_t * inbuf_list; //a free list contain all recv data
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
    
    //printf("[%d]: adapt reduce generic\n", rank);
    
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_reduce_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_reduce_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //not leaf
    if (tree->tree_nextsize > 0) {
        inbuf_list = OBJ_NEW(opal_free_list_t);
        opal_free_list_init(inbuf_list,
                            sizeof(mca_coll_adapt_inbuf_t) + real_seg_size,
                            opal_cache_line_size,
                            OBJ_CLASS(mca_coll_adapt_inbuf_t),
                            0,opal_cache_line_size,
                            FREE_LIST_NUM_INBUF_LIST,
                            FREE_LIST_MAX_INBUF_LIST,
                            FREE_LIST_INC_INBUF_LIST,
                            NULL, 0, NULL, NULL, NULL);
        //set up next_recv_segs
        next_recv_segs = (int32_t *)malloc(sizeof(int32_t) * tree->tree_nextsize);
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
    
    //create recv_list
    recv_list = OBJ_NEW(opal_list_t);
    
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_reduce_context_t *con = OBJ_NEW(mca_coll_adapt_constant_reduce_context_t);
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
    con->inbuf_list = inbuf_list;
    con->recv_list = recv_list;
    con->lower_bound = lower_bound;
    con->ongoing_send = 0;
    
    // non leaf nodes
    if (tree->tree_nextsize > 0) {
        //set accumbuf
        if (root == rank){
            accumbuf = (char*)rbuf;
            if (sbuf != MPI_IN_PLACE) {
                ompi_datatype_copy_content_same_ddt(dtype, count, (char*)accumbuf, (char*)sbuf);
            }
        }
        else{
            accumbuf_free = (char*)malloc(true_extent +
                                          (ptrdiff_t)(count - 1) * extent);
            accumbuf = accumbuf_free - lower_bound;
            //copy sbuf to accumbuf
            ompi_datatype_copy_content_same_ddt(dtype, count, (char*)accumbuf, (char*)sbuf);
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
                    //get inbuf
                    //printf("wait inbuf_list\n");
                    mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
                    //printf("sizeof(inbuf_t) = %d, real_seg_size = %d\n", sizeof(mca_coll_adapt_inbuf_t), real_seg_size);
                    //get context
                    //printf("wait context_list\n");
                    mca_coll_adapt_reduce_context_t * context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(context_list);
                    context->buff = NULL;
                    context->frag_id = seg_index;
                    context->child_id = i;              //the id of peer in in the tree
                    context->peer = tree->tree_next[i];   //the actural rank of the peer
                    context->con = con;
                    OBJ_RETAIN(con);
                    context->inbuf = inbuf;
                    
                    //printf("[%d]: In reduce, create irecv for seg %d, peer %d, recv_count %d\n", context->con->rank, context->frag_id, context->peer, recv_count);
                    
                    //create a recv request
                    ompi_request_t *recv_req;
                    err = MCA_PML_CALL(irecv(inbuf->buff - lower_bound, recv_count, dtype, tree->tree_next[i], seg_index, comm, &recv_req));
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
        
    }
    
    //leaf nodes
    else{
        mca_coll_adapt_item_t *item;
        //set up recv_list
        for(seg_index = 0; seg_index < num_segs; seg_index++) {
            item = OBJ_NEW(mca_coll_adapt_item_t);
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
                //printf("wait context_list\n");
                mca_coll_adapt_reduce_context_t * context = (mca_coll_adapt_reduce_context_t *) opal_free_list_wait(context_list);
                context->buff = (char*)sbuf + (ptrdiff_t)item->id * (ptrdiff_t)segment_increment;
                context->frag_id = item->id;
                context->peer = tree->tree_prev;   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                //atomic
                opal_atomic_add_32(&(context->con->ongoing_send), 1);
                //int* ta1 = (int *)((char*)sbuf + (ptrdiff_t)item->id * (ptrdiff_t)segment_increment);
                //printf("[%d]: In reduce, create isend to seg %d, peer %d, send_count %d, sbuf contains %d, %d\n", context->con->rank, context->frag_id, context->peer, ta1[0], ta1[1]);
                
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
    
    if (accumbuf_free != NULL) {
        free(accumbuf_free);
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
        OBJ_RELEASE(inbuf_list);
        free(next_recv_segs);
    }
    OBJ_RELEASE(context_list);
    return MPI_SUCCESS;
}



