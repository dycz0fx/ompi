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
#include "opal/util/bit_ops.h"      //opal_next_poweroftwo

#define FREE_LIST_NUM_CONTEXT_LIST 10    //The start size of the context free list
#define FREE_LIST_MAX_CONTEXT_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_CONTEXT_LIST 10    //The incresment of the context free list
#define FREE_LIST_NUM_INBUF_LIST 2    //The start size of the context free list
#define FREE_LIST_MAX_INBUF_LIST 10000  //The max size of the context free list
#define FREE_LIST_INC_INBUF_LIST 2    //The incresment of the context free list

#define TEST printfno

static void printfno(){
    
}

int mca_coll_adapt_allreduce_intra_nonoverlapping(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    TEST("In adapt allreduce_intra_nonoverlapping\n");
    int err, rank;
    rank = ompi_comm_rank(comm);
    
    /* Reduce to 0 and broadcast. */
    if (MPI_IN_PLACE == sbuf) {
        if (0 == rank) {
            err = comm->c_coll.coll_reduce (MPI_IN_PLACE, rbuf, count, dtype, op, 0, comm, comm->c_coll.coll_reduce_module);
        } else {
            err = comm->c_coll.coll_reduce (rbuf, NULL, count, dtype, op, 0, comm, comm->c_coll.coll_reduce_module);
        }
    } else {
        err = comm->c_coll.coll_reduce (sbuf, rbuf, count, dtype, op, 0, comm, comm->c_coll.coll_reduce_module);
    }
    if (MPI_SUCCESS != err) {
        return err;
    }
    
    return comm->c_coll.coll_bcast (rbuf, count, dtype, 0, comm, comm->c_coll.coll_bcast_module);
}

static int send_cb(ompi_request_t *req);
static int recv_cb(ompi_request_t *req);

static int send_cb(ompi_request_t *req){
    
    req->req_complete_cb_called = 1;
    
    mca_coll_adapt_allreduce_context_t *context = (mca_coll_adapt_allreduce_context_t *) req->req_complete_cb_data;
    
    TEST("[%d]: send_cb, peer = %d, distance = %d, inbuf_ready = %d, sendbuf_ready = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->distance, context->con->inbuf_ready, context->con->sendbuf_ready);
    int err;
    int rank = ompi_comm_rank(context->con->comm);
    //set new distance
    int new_distance = 0;
    if (context->distance == 0) {
        new_distance = 1;
    }
    else{
        new_distance = context->distance << 1;
    }
    OPAL_THREAD_LOCK(context->con->mutex_buf);
    context->con->sendbuf_ready++;
    int ready = context->con->sendbuf_ready && context->con->inbuf_ready;
    OPAL_THREAD_UNLOCK(context->con->mutex_buf);
    if (ready) {
        opal_atomic_add_32(&(context->con->sendbuf_ready), -1);
        opal_atomic_add_32(&(context->con->inbuf_ready), -1);
        
        int newremote = 0;
        int remote = 0;
        mca_coll_adapt_allreduce_context_t * recv_context = NULL;
        //recv from new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = context->newrank;
            recv_context->distance = new_distance;
            newremote = context->newrank ^ new_distance;
            remote = (newremote < context->con->extra_ranks)?(newremote * 2 + 1):(newremote + context->con->extra_ranks);
            recv_context->peer = remote;
            recv_context->con = context->con;
            OBJ_RETAIN(recv_context->con);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in send cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(inbuf->buff-context->con->lower_bound, recv_context->con->count, recv_context->con->datatype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
                recv_cb(recv_req);
            }
        }
        //do the operation, commutative
        char* recvbuf;
        if (context->inbuf != NULL) {
            recvbuf = context->inbuf->buff-context->con->lower_bound;
        }
        else{
            recvbuf = context->con->recvbuf;
        }
        //sendbuf = recvbuf + sendbuf
        ompi_op_reduce(context->con->op, recvbuf, context->con->sendbuf, context->con->count, context->con->datatype);
        
        //send to new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->inbuf = recv_context->inbuf;
            send_context->newrank = context->newrank;
            send_context->distance = new_distance;
            send_context->peer = remote;
            send_context->con = context->con;
            OBJ_RETAIN(send_context->con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in send cb): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
            err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                send_cb(send_req);
            }
        }
        
        //this is the last send
        if (new_distance >= context->con->adjsize){
            if (context->newrank >=0) {
                //at last, send to rank - 1
                if (rank < (2 * context->con->extra_ranks) && rank % 2 == 1) {
                    mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
                    send_context->inbuf = NULL;
                    send_context->newrank = context->newrank;
                    send_context->distance = context->con->adjsize+1;
                    send_context->peer = rank-1;
                    send_context->con = context->con;
                    OBJ_RETAIN(send_context->con);
                    //set new_distance, so in this turn would not enter the complete part
                    new_distance = context->distance;
                    //create a send request
                    ompi_request_t *send_req;
                    TEST("[%d]: Send(start in send cb, Last): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
                    err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke send call back
                    if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                        send_cb(send_req);
                    }
                }
                //copy to recvbuf
                ompi_datatype_copy_content_same_ddt(context->con->datatype, context->con->count, context->con->recvbuf, context->con->sendbuf);
            }
        }
    }

    int * temp_total_send = &(context->con->total_send);
    TEST("adjsize %d, new_distance %d, new_rank %d, total_send %d\n",context->con->adjsize, new_distance, context->newrank, context->con->total_send);
    //this is the last send the node with newrank < 0 only do one send
    if (context->con->total_send == 1) {
        int complete;
        complete = opal_atomic_add_32(&(context->con->complete), 1);

        TEST("[%d]: last send, complete = %d, total_send = %d\n", ompi_comm_rank(context->con->comm), complete, context->con->total_send);
        if (complete == 2) {
            //signal
            TEST("[%d]: last send, signal\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            opal_free_list_t * temp = context->con->context_list;
            OBJ_RELEASE(context->con->inbuf_list);
            OBJ_RELEASE(context->con->context_list);
            free(context->con->sendbuf);
            OBJ_RELEASE(context->con->mutex_buf);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OPAL_THREAD_LOCK(&ompi_request_lock);
            ompi_request_complete(temp_req, 1);
            OPAL_THREAD_UNLOCK(&ompi_request_lock);
        }
    }
    else{
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        opal_atomic_add_32(temp_total_send, -1);
    }
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: send_cb finish\n", rank);
    return 1;
}

static int recv_cb(ompi_request_t *req){
    
    req->req_complete_cb_called = 1;

    mca_coll_adapt_allreduce_context_t *context = (mca_coll_adapt_allreduce_context_t *) req->req_complete_cb_data;

    TEST("[%d]: recv_cb, peer = %d, distance = %d, inbuf_ready = %d, sendbuf_ready = %d\n", ompi_comm_rank(context->con->comm), context->peer, context->distance, context->con->inbuf_ready, context->con->sendbuf_ready);
    
    int err;
    int rank = ompi_comm_rank(context->con->comm);

    //set new distance
    int new_distance = 0;
    if (context->distance == 0) {
        new_distance = 1;
    }
    else{
        new_distance = context->distance << 1;
    }
    
    OPAL_THREAD_LOCK(context->con->mutex_buf);
    context->con->inbuf_ready++;
    int ready = context->con->sendbuf_ready && context->con->inbuf_ready;
    OPAL_THREAD_UNLOCK(context->con->mutex_buf);
    if (ready) {
        opal_atomic_add_32(&(context->con->sendbuf_ready), -1);
        opal_atomic_add_32(&(context->con->inbuf_ready), -1);

        int newremote;
        int remote;
        mca_coll_adapt_allreduce_context_t * recv_context;
        //recv from new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(context->con->inbuf_list);
            recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = context->newrank;
            recv_context->distance = new_distance;
            newremote = context->newrank ^ new_distance;
            remote = (newremote < context->con->extra_ranks)?(newremote * 2 + 1):(newremote + context->con->extra_ranks);
            recv_context->peer = remote;
            recv_context->con = context->con;
            OBJ_RETAIN(recv_context->con);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in recv cb): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(inbuf->buff-context->con->lower_bound, recv_context->con->count, recv_context->con->datatype, recv_context->peer, recv_context->distance, recv_context->con->comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
                recv_cb(recv_req);
            }
        }
        //do the operation, commutative
        char* recvbuf;
        if (context->inbuf != NULL) {
            recvbuf = context->inbuf->buff-context->con->lower_bound;
        }
        else{
            recvbuf = context->con->recvbuf;
        }
        //sendbuf = recvbuf + sendbuf
        ompi_op_reduce(context->con->op, recvbuf, context->con->sendbuf, context->con->count, context->con->datatype);
        
        //send to new distance
        if (new_distance < context->con->adjsize && context->newrank >= 0) {
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->inbuf = recv_context->inbuf;
            send_context->newrank = context->newrank;
            send_context->distance = new_distance;
            send_context->peer = remote;
            send_context->con = context->con;
            OBJ_RETAIN(send_context->con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in recv cb): distance %d to %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                send_cb(send_req);
            }
        }
        
        //this is the last recv
        if (new_distance >= context->con->adjsize){
            if (context->newrank >=0) {
                //at last, send to rank - 1
                if (rank < (2 * context->con->extra_ranks) && rank % 2 == 1) {
                    mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context->con->context_list);
                    send_context->inbuf = NULL;
                    send_context->newrank = context->newrank;
                    send_context->distance = context->con->adjsize+1;
                    send_context->peer = rank-1;
                    send_context->con = context->con;
                    OBJ_RETAIN(send_context->con);
                    
                    //create a send request
                    ompi_request_t *send_req;
                    TEST("[%d]: Send(start in recv cb, Last): distance %d to %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer);
                    err = MCA_PML_CALL(isend(send_context->con->sendbuf, send_context->con->count, send_context->con->datatype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke send call back
                    if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                        send_cb(send_req);
                    }
                }
                //copy to recvbuf
                ompi_datatype_copy_content_same_ddt(context->con->datatype, context->con->count, context->con->recvbuf, context->con->sendbuf);
            }
        }
    }
    
    int * temp_total_recv = &(context->con->total_recv);
    //this is the last recv, the node with newrank < 0 only do one recv
    if (context->con->total_recv == 1){
        int complete = opal_atomic_add_32(&(context->con->complete), 1);
        TEST("[%d]: last recv, complete = %d\n", ompi_comm_rank(context->con->comm), complete);
        if (complete == 2) {
            //signal
            TEST("[%d]: last recv, signal\n", ompi_comm_rank(context->con->comm));
            ompi_request_t *temp_req = context->con->request;
            if (context->newrank > 0) {
                opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
            }
            opal_free_list_t * temp = context->con->context_list;
            OBJ_RELEASE(context->con->inbuf_list);
            OBJ_RELEASE(context->con->context_list);
            free(context->con->sendbuf);
            OBJ_RELEASE(context->con->mutex_buf);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OPAL_THREAD_LOCK(&ompi_request_lock);
            ompi_request_complete(temp_req, 1);
            OPAL_THREAD_UNLOCK(&ompi_request_lock);
        }
    }
    else{
        if (context->newrank > 0) {
            opal_free_list_return(context->con->inbuf_list, (opal_free_list_item_t*)context->inbuf);
        }
        opal_free_list_t * temp = context->con->context_list;
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        opal_atomic_add_32(temp_total_recv, -1);
    }
    OPAL_THREAD_UNLOCK (req->req_lock);
    req->req_free(&req);
    TEST("[%d]: recv_cb finish\n", rank);
    return 1;
}

int mca_coll_adapt_allreduce_intra_recursivedoubling(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    TEST("In adapt allreduce_intra_recursivedoubling\n");
    ptrdiff_t extent, lower_bound, true_lower_bound, true_extent;
    int size, rank, adjsize, extra_ranks;
    char *accumbuf = NULL;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    int err;
    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
        }
        return MPI_SUCCESS;
    }
    
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    ompi_datatype_get_true_extent(dtype, &true_lower_bound, &true_extent);

    /* Allocate and initialize temporary send buffer */
    accumbuf = (char*) malloc(true_extent + (ptrdiff_t)(count - 1) * extent);
    if (MPI_IN_PLACE == sbuf) {
        ompi_datatype_copy_content_same_ddt(dtype, count, accumbuf, (char*)rbuf);
    }
    else{
        ompi_datatype_copy_content_same_ddt(dtype, count, accumbuf, (char*)sbuf);
    }
    
    /* Determine nearest power of two less than or equal to size */
    /* size = 10, adjsize =16 */
    adjsize = opal_next_poweroftwo (size);
    adjsize >>= 1;
    extra_ranks = size - adjsize;

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

    //set up free list
    opal_free_list_t * context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_allreduce_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_allreduce_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_CONTEXT_LIST,
                        FREE_LIST_MAX_CONTEXT_LIST,
                        FREE_LIST_INC_CONTEXT_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    opal_free_list_t * inbuf_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(inbuf_list,
                        sizeof(mca_coll_adapt_inbuf_t) + true_extent + (ptrdiff_t)(count - 1) * extent,
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_inbuf_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM_INBUF_LIST,
                        FREE_LIST_MAX_INBUF_LIST,
                        FREE_LIST_INC_INBUF_LIST,
                        NULL, 0, NULL, NULL, NULL);
    
    //set up mutex
    opal_mutex_t * mutex_buf = OBJ_NEW(opal_mutex_t);
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_allreduce_context_t *con = OBJ_NEW(mca_coll_adapt_constant_allreduce_context_t);
    con->sendbuf = accumbuf;
    con->recvbuf = rbuf;
    con->count = count;
    con->datatype = dtype;
    con->comm = comm;
    con->request = temp_request;
    con->context_list = context_list;
    con->op = op;
    con->lower_bound = lower_bound;
    con->extra_ranks = extra_ranks;
    con->inbuf_list = inbuf_list;
    con->complete = 0;
    con->adjsize = adjsize;
    con->sendbuf_ready = 0;     //use to decide if sendbuf is ready for reuse
    con->total_send = 0;         //to tell how many sends are needed in total
    con->total_recv = 0;        //to tell how many recvs are needed in total
    con->mutex_buf = mutex_buf;
    
    /* Handle non-power-of-two case:
     - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
     sets new rank to -1.
     - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
     apply appropriate operation, and set new rank to rank/2
     - Everyone else sets rank to rank - extra_ranks
     Turn non-power-of-two case into power of two case to improve performance.
     Suppose size = 2^n + extra_ranks. By combining every pair of 2 nodes among 2 * extra_ranks of nodes into one node, 2*extra_ranks nodes become extra_ranks node.
     So the size become 2^n. The goal is to remove the extra_ranks number of nodes.
     */
    if (rank <  (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            TEST("[%d]: Case 1\n", rank);
            con->total_send = 1;
            con->total_recv = 1;
            int newrank = -1;
            //send to rank+1
            mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
            send_context->inbuf = NULL;
            send_context->newrank = newrank;
            send_context->distance = 0;
            send_context->peer = rank+1;
            send_context->con = con;
            OBJ_RETAIN(con);
            
            //create a send request
            ompi_request_t *send_req;
            TEST("[%d]: Send(start in main): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);
            err = MCA_PML_CALL(isend(con->sendbuf, count, dtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke send call back
            if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
                send_cb(send_req);
            }
            
            //recv from rank+1 at last round, since this node just recv once at last,
            //so there is no need to use inbuf_list, set distance to adjsize+1
            mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
            recv_context->inbuf = NULL;
            recv_context->newrank = newrank;
            recv_context->distance = adjsize+1;
            recv_context->peer = rank+1;
            recv_context->con = con;
            OBJ_RETAIN(con);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(con->recvbuf, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
                recv_cb(recv_req);
            }
        }
        else {
            TEST("[%d]: Case 2\n", rank);
            con->total_send = log2_int(adjsize)+1;
            con->total_recv = con->total_send;
            int newrank = rank>>1;
            //recv from rank-1
            mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
            mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
            recv_context->inbuf = inbuf;
            recv_context->newrank = newrank;
            recv_context->distance = 0;
            recv_context->peer = rank-1;
            recv_context->con = con;
            OBJ_RETAIN(con);
            //there is no send going, sendbuf is ready
            opal_atomic_add_32(&(recv_context->con->sendbuf_ready), 1);
            //create a recv request
            ompi_request_t *recv_req;
            TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
            err = MCA_PML_CALL(irecv(inbuf->buff-lower_bound, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke recv call back
            if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
                recv_cb(recv_req);
            }
        }
    }
    else {
        TEST("[%d]: Case 3\n", rank);
        con->total_send = log2_int(adjsize);
        con->total_recv = con->total_send;
        int newrank = rank-extra_ranks;
        int newremote;
        int remote;
        //recv from distance = 1
        mca_coll_adapt_inbuf_t * inbuf = (mca_coll_adapt_inbuf_t *) opal_free_list_wait(inbuf_list);
        mca_coll_adapt_allreduce_context_t * recv_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
        recv_context->inbuf = inbuf;
        recv_context->newrank = newrank;
        recv_context->distance = 1;
        /* Determine remote node */
        newremote = recv_context->newrank ^ recv_context->distance;
        remote = (newremote < extra_ranks)?(newremote * 2 + 1):(newremote + extra_ranks);
        recv_context->peer = remote;
        recv_context->con = con;
        OBJ_RETAIN(con);
        //create a recv request
        ompi_request_t *recv_req;
        TEST("[%d]: Recv(start in main): distance %d from %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->distance, recv_context->peer);
        err = MCA_PML_CALL(irecv(inbuf->buff-lower_bound, count, dtype, recv_context->peer, recv_context->distance, comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke recv call back
        if(!ompi_request_set_callback(recv_req, recv_cb, recv_context)) {
            recv_cb(recv_req);
        }
        
        //send to distance = 1
        mca_coll_adapt_allreduce_context_t * send_context = (mca_coll_adapt_allreduce_context_t *) opal_free_list_wait(context_list);
        send_context->inbuf = recv_context->inbuf;
        send_context->newrank = newrank;
        send_context->distance = 1;
        send_context->peer = remote;
        send_context->con = con;
        OBJ_RETAIN(con);
        
        //create a send request
        ompi_request_t *send_req;
        TEST("[%d]: Send(start in main): distance %d to %d, ongoing send %d\n", ompi_comm_rank(send_context->con->comm), send_context->distance, send_context->peer, send_context->con->total_send);

        err = MCA_PML_CALL(isend(con->sendbuf, count, dtype, send_context->peer, send_context->distance, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        if(!ompi_request_set_callback(send_req, send_cb, send_context)) {
            send_cb(send_req);
        }
        
    }
    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    
    return MPI_SUCCESS;
}

int mca_coll_adapt_allreduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    return mca_coll_adapt_allreduce_intra_recursivedoubling(sbuf, rbuf, count, dtype, op, comm, module);
}



