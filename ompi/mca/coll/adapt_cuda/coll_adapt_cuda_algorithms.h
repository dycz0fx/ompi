#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"  //ompi_coll_tree_t

int mca_coll_adapt_cuda_bcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_nccl(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_two_chains(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_bcast_nccl_topo(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t *tree);

int mca_coll_adapt_cuda_bcast_generic(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t *tree);

int mca_coll_adapt_cuda_bcast_generic_cpu(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t *tree);

int mca_coll_adapt_cuda_bcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t** trees);

int mca_coll_adapt_cuda_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_bininary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);


int mca_coll_adapt_cuda_ibcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_two_chains(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_cuda_ibcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t** trees);

int mca_coll_adapt_cuda_reduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_topoaware_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_topoaware_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_reduce_chain_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_cuda_reduce_topo_generic( const void* sendbuf, void* recvbuf, int original_count,
                                    struct ompi_datatype_t* datatype, struct ompi_op_t* op,
                                    int root, struct ompi_communicator_t* comm,
                                    mca_coll_base_module_t *module,
                                    ompi_coll_tree_t* tree, int count_by_segment,
                                    int max_outstanding_reqs );

int mca_coll_adapt_cuda_reduce_topo_generic_cpu( const void* sendbuf, void* recvbuf, int original_count,
                                    struct ompi_datatype_t* datatype, struct ompi_op_t* op,
                                    int root, struct ompi_communicator_t* comm,
                                    mca_coll_base_module_t *module,
                                    ompi_coll_tree_t* tree, int count_by_segment,
                                    int max_outstanding_reqs );

int mca_coll_adapt_cuda_reduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_cuda_ireduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_in_order_binomial(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_binary(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_pipeline(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);


int mca_coll_adapt_cuda_ireduce_chain(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_linear(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ireduce_generic(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree);

int mca_coll_adapt_cuda_allreduce_intra_nonoverlapping(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_allreduce_intra_recursivedoubling(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_allreduce(const void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_iallreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_alltoallv(const void *sbuf, const int *scounts, const int *sdisps, struct ompi_datatype_t *sdtype, void* rbuf, const int *rcounts, const int *rdisps, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_adapt_cuda_ialltoallv(const void *sbuf, const int *scounts, const int *sdisps, struct ompi_datatype_t *sdtype, void* rbuf, const int *rcounts, const int *rdisps, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);

int coll_adapt_cuda_bcast_progress();
//get thread id for test
static inline uint64_t gettid(void) {
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
