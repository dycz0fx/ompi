#include "coll_adapt_algorithms.h"

int mca_coll_adapt_iallreduce(void *sbuf, void *rbuf, int count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    printf("In adapt iallreduce\n");
    return MPI_SUCCESS;
    
}
