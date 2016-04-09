#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"  //ompi_coll_tree_t

int mca_coll_adapt_bcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_bcast_generic(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t *tree);

int mca_coll_adapt_bcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t** trees);

int mca_coll_adapt_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_bininary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_ibcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t** trees);

int mca_coll_adapt_reduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_binomial(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_in_order_binomial(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_binary(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_pipeline(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_chain(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_linear(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_reduce_generic(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_ireduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_binomial(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_in_order_binomial(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_binary(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_pipeline(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);


int mca_coll_adapt_ireduce_chain(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_linear(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_ireduce_generic(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_allreduce_intra_nonoverlapping(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_allreduce_intra_recursivedoubling(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_allreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_iallreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

//get thread id for test
static inline uint64_t gettid() {
    pthread_t ptid = pthread_self();
    uint64_t threadId = 0;
    int min;
    if (sizeof(threadId) < sizeof(ptid)) {
        min = sizeof(threadId);
    }
    else
        min = sizeof(ptid);
    memcpy(&threadId, &ptid, min);
    return threadId;
}

//print tree for test
static inline void print_tree(ompi_coll_tree_t* tree, int rank) {
    int i;
    printf("[%d, prev = %d, next_size = %d]:", rank, tree->tree_prev, tree->tree_nextsize);
    for( i = 0; i < tree->tree_nextsize; i++ ){
        printf(" %d", tree->tree_next[i]);
    }
    if (rank == tree->tree_root) {
        printf(" root = %d", tree->tree_root);
    }
    printf("\n");
}
            
static inline int adapt_request_free(ompi_request_t** request)
{
    (*request)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*request);
    *request = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}
