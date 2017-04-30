#ifndef MCA_COLL_ADAPT_CUDA_TOPO_H_HAS_BEEN_INCLUDED
#define MCA_COLL_ADAPT_CUDA_TOPO_H_HAS_BEEN_INCLUDED

#include "ompi_config.h"
#include "ompi/mca/coll/base/coll_base_topo.h" 

BEGIN_C_DECLS

typedef struct ompi_coll_adapt_cuda_tree_t {
    ompi_coll_tree_t super;
    uint32_t topo_flags;
} ompi_coll_adapt_cuda_tree_t;

ompi_coll_adapt_cuda_tree_t*
ompi_coll_adapt_cuda_topo_build_tree_chain_intra_node( struct ompi_communicator_t* comm,
                                                       int root );

END_C_DECLS

#endif  /* MCA_COLL_ADAPT_CUDA_TOPO_H_HAS_BEEN_INCLUDED */