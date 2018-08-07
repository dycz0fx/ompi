#include "coll_adapt.h"
#include "coll_adapt_context.h"

/* Bcast algorithm variables */
static int coll_adapt_ibcast_algorithm = 0;
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
};

int mca_coll_adapt_ibcast_init(void){
    mca_base_component_t *c = &mca_coll_adapt_component.super.collm_version;
    
    mca_base_component_var_register(c, "bcast_algorithm",
                                    "Algorithm of broadcast",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &coll_adapt_ibcast_algorithm);
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_fini(void){
    if (NULL != coll_adapt_ibcast_context_free_list) {
        OBJ_RELEASE(coll_adapt_ibcast_context_free_list);
        coll_adapt_ibcast_context_free_list = NULL;
        coll_adapt_ibcast_context_free_list_enabled = 0;
        OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "bcast fini\n"));
    }
    return OMPI_SUCCESS;
}

static mca_coll_adapt_bcast_context_t * ibcast_init_context(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t *request, opal_mutex_t *mutex, ompi_coll_tree_t *tree, int ibcast_tag, int peer, int *num_sent, int *num_recv){
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(coll_adapt_ibcast_context_free_list);
    context->buff = buff;
    context->root = root;
    context->count = count;
    context->datatype = datatype;
    context->comm = comm;
    context->request = request;
    context->mutex = mutex;
    context->tree = tree;
    context->ibcast_tag = ibcast_tag;
    context->peer = peer;
    context->num_sent = num_sent;
    context->num_recv = num_recv;
    return context;
}

static int ibcast_fini_context(){
    return OMPI_SUCCESS;
}

static int ibcast_request_fini(mca_coll_adapt_bcast_context_t *context)
{
    ompi_request_t *temp_req = context->request;
    OBJ_RELEASE(context->mutex);
    free(context->num_sent);
    free(context->num_recv);
    opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;
}

/* send call back */
static int send_cb(ompi_request_t *req){
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    int rank = ompi_comm_rank(context->comm);
    
    /* for testing output the log message */
    //opal_output_init();
    //mca_pml_ob1_dump(context->con->comm, 0);
    //opal_output_finalize();
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(cb): to %d at buff %p root %d\n", rank, gettid(), context->peer, (void *)context->buff, context->root));
    
    OPAL_THREAD_LOCK(context->mutex);
    int num_sent = ++(*(context->num_sent));
    int num_recv = *(context->num_recv);
    opal_mutex_t * mutex_temp = context->mutex;
    /* check whether complete the request */
    if ((rank == context->root && num_sent == context->tree->tree_nextsize) ||
        (context->tree->tree_nextsize > 0 && rank != context->root && num_sent == context->tree->tree_nextsize && num_recv == 1) ||
        (context->tree->tree_nextsize == 0 && num_recv == 1)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Complete in send\n", rank));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else {
        opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

/* receive call back */
static int recv_cb(ompi_request_t *req){
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    int rank = ompi_comm_rank(context->comm);
    int err, i;
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(cb): from %d at buff %p root %d\n", ompi_comm_rank(context->comm), gettid(), context->peer, (void *)context->buff, context->root));
    
    OPAL_THREAD_LOCK(context->mutex);
    /* send to its children */
    for (i = 0; i < context->tree->tree_nextsize; i++) {
        mca_coll_adapt_bcast_context_t * send_context = ibcast_init_context(context->buff, context->count, context->datatype, context->root, context->comm, context->request, context->mutex, context->tree, context->ibcast_tag, context->tree->tree_next[i], context->num_sent, context->num_recv);
        ompi_request_t *send_req;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Send(start in recv cb): to %d at buff %p send_count %d tag %d\n", ompi_comm_rank(send_context->comm), send_context->peer, send_context->buff, send_context->count, send_context->ibcast_tag));
        err = MCA_PML_CALL(isend(send_context->buff, send_context->count, send_context->datatype, send_context->peer, send_context->ibcast_tag, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->comm, &send_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        //invoke send call back
        OPAL_THREAD_UNLOCK(context->mutex);
        ompi_request_set_callback(send_req, send_cb, send_context);
        OPAL_THREAD_LOCK(context->mutex);
    }
    
    int num_sent = *(context->num_sent);
    int num_recv = ++(*(context->num_recv));
    opal_mutex_t * mutex_temp = context->mutex;
    //check whether signal the condition
    if ((rank == context->root && num_sent == context->tree->tree_nextsize) ||
        (context->tree->tree_nextsize > 0 && rank != context->root && num_sent == context->tree->tree_nextsize && num_recv == 1) ||
        (context->tree->tree_nextsize == 0 && num_recv == 1)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Complete in recv\n", ompi_comm_rank(context->comm)));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else{
        opal_free_list_return(coll_adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}


int mca_coll_adapt_ibcast_intra(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    printf("adapt\n");
    /* if count == 0, just return a completed request */
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
            OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "ibcast root %d, algorithm %d\n", root, coll_adapt_ibcast_algorithm));
        }
        int ibcast_tag = opal_atomic_add_fetch_32((&(comm->c_ibcast_tag)), 1);
        mca_coll_adapt_ibcast_fn_t bcast_func = (mca_coll_adapt_ibcast_fn_t)mca_coll_adapt_ibcast_algorithm_index[coll_adapt_ibcast_algorithm].algorithm_fn_ptr;
        return bcast_func(buff, count, datatype, root, comm, request, module, ibcast_tag);
        //return mca_coll_adapt_ibcast_binomial(buff, count, datatype, root, comm, request, module, ibcast_tag);
    }
}

int mca_coll_adapt_ibcast_tuned(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "tuned\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "binomial\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "in_order_binomial\n"));
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_in_order_bmtree) && (coll_comm->cached_in_order_bmtree_root == root) ) ) {
        if( coll_comm->cached_in_order_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_in_order_bmtree) );
        }
        coll_comm->cached_in_order_bmtree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
        coll_comm->cached_in_order_bmtree_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_in_order_bmtree, ibcast_tag);
}

int mca_coll_adapt_ibcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "binary\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "pipeline\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "chain\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "linear\n"));
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, int ibcast_tag){
    int i;          /* temp variable for iteration */
    int size;       /* size of the communicator */
    int rank;       /* rank of this node */
    int err;        /* record return value */
    
    ompi_request_t * temp_request = NULL;  /* the request be passed outside */
    opal_mutex_t * mutex;
    
    /* set up free list if needed*/
    if (0 == coll_adapt_ibcast_context_free_list_enabled) {
        int32_t context_free_list_enabled = opal_atomic_add_fetch_32((&(coll_adapt_ibcast_context_free_list_enabled)), 1);
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
    int *num_sent = (int *)malloc(sizeof(int));
    *num_sent = 0;
    int *num_recv = (int *)malloc(sizeof(int));
    *num_recv = 0;

    OPAL_THREAD_LOCK(mutex);
    /* if root, send segment to every children */
    if (rank == root){
        for (i=0; i<tree->tree_nextsize; i++) {
            mca_coll_adapt_bcast_context_t * context = ibcast_init_context(buff, count, datatype, root, comm, temp_request, mutex, tree, ibcast_tag, tree->tree_next[i], num_sent, num_recv);
            ompi_request_t *send_req;
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(start in main): to %d at buff %p count %d tag %d\n", rank, gettid(), context->peer, (void *)buff, count, ibcast_tag));
            err = MCA_PML_CALL(isend(buff, count, datatype, context->peer, ibcast_tag, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            /* invoke send call back */
            OPAL_THREAD_UNLOCK(mutex);
            ompi_request_set_callback(send_req, send_cb, context);
            OPAL_THREAD_LOCK(mutex);
        }
    }
    
    /* if not root, receive data from parent in the tree */
    else{
        mca_coll_adapt_bcast_context_t * context = ibcast_init_context(buff, count, datatype, root, comm, temp_request, mutex, tree, ibcast_tag, tree->tree_prev, num_sent, num_recv);
        ompi_request_t *recv_req;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(start in main): from %d at buff %p count %d tag %d\n", ompi_comm_rank(context->comm), gettid(), context->peer, buff, count, ibcast_tag));
        err = MCA_PML_CALL(irecv(buff, count, datatype, context->peer, ibcast_tag, comm, &recv_req));
        if (MPI_SUCCESS != err) {
            return err;
        }
        /* invoke receive call back */
        OPAL_THREAD_UNLOCK(mutex);
        ompi_request_set_callback(recv_req, recv_cb, context);
        OPAL_THREAD_LOCK(mutex);
    }
    
    OPAL_THREAD_UNLOCK(mutex);
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: End of Ibcast\n", rank, gettid()));
    
    return MPI_SUCCESS;
}



