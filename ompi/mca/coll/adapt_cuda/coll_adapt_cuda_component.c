/*
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#include "ompi_config.h"

#include "opal/util/show_help.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "coll_adapt_cuda.h"
#include "coll_adapt_cuda_mpool.h"
#include "coll_adapt_cuda_nccl.h"
#include "coll_adapt_cuda_algorithms.h"
#include "coll_adapt_cuda_context.h"
#include "opal/mca/common/cuda/common_cuda.h"


/*
 * Public string showing the coll ompi_adapt component version number
 */
const char *mca_coll_adapt_cuda_component_version_string =
    "Open MPI ADAPT CUDA collective MCA component version " OMPI_VERSION;


/*
 * Local functions
 */
static int coll_adapt_cuda_progress();
static int adapt_cuda_open(void);
static int adapt_cuda_close(void);
static int adapt_cuda_register(void);

static int coll_adapt_cuda_shared_mem_used_data;

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_adapt_cuda_component_t mca_coll_adapt_cuda_component = {

    /* First, fill in the super */

    {
        /* First, the mca_component_t struct containing meta
           information about the component itself */
        
        {
            MCA_COLL_BASE_VERSION_2_0_0,

            /* Component name and version */
            "adapt_cuda",
            OMPI_MAJOR_VERSION,
            OMPI_MINOR_VERSION,
            OMPI_RELEASE_VERSION,

            /* Component functions */
            adapt_cuda_open, /* open */
            adapt_cuda_close,
            NULL, /* query */
            adapt_cuda_register
        },
        {
            /* The component is not checkpoint ready */
            MCA_BASE_METADATA_PARAM_NONE
        },

        /* Initialization / querying functions */
        
        mca_coll_adapt_cuda_init_query,
        mca_coll_adapt_cuda_comm_query,
    },

    /* adapt-component specific information */

    /* (default) priority */
    0,

    /* (default) control size (bytes) */
    4096,

    /* (default) number of "in use" flags for each communicator's area
       in the per-communicator shmem segment */
    2,

    /* (default) number of segments for each communicator in the
       per-communicator shmem segment */
    8,

    /* (default) fragment size */
    8192,

    /* (default) degree of tree for tree-based operations (must be <=
       control unit size) */
    4,

    /* (default) number of processes in coll_adapt_cuda_shared_mem_size
       information variable */
    4,

    /* default values for non-MCA parameters */
    /* Not specifying values here gives us all 0's */
};

/* open the component */
static int adapt_cuda_open(void)
{
    int rc;
    rc = opal_progress_register(coll_adapt_cuda_progress);
    if (OMPI_SUCCESS != rc ) {
        fprintf(stderr," failed to register the ml progress function \n");
        fflush(stderr);
        return rc;
    }
    return OMPI_SUCCESS;
}

/*
 * Shut down the component
 */
static int adapt_cuda_close(void)
{
    return OMPI_SUCCESS;
}

static int adapt_cuda_verify_mca_variables(void)
{
    mca_coll_adapt_cuda_component_t *cs = &mca_coll_adapt_cuda_component;

    if (0 != (cs->adapt_fragment_size % cs->adapt_control_size)) {
        cs->adapt_fragment_size += cs->adapt_control_size - 
            (cs->adapt_fragment_size % cs->adapt_control_size);
    }

    if (cs->adapt_comm_num_in_use_flags < 2) {
        cs->adapt_comm_num_in_use_flags = 2;
    }

    if (cs->adapt_comm_num_segments < cs->adapt_comm_num_in_use_flags) {
        cs->adapt_comm_num_segments = cs->adapt_comm_num_in_use_flags;
    }
    if (0 != (cs->adapt_comm_num_segments % cs->adapt_comm_num_in_use_flags)) {
        cs->adapt_comm_num_segments += cs->adapt_comm_num_in_use_flags - 
            (cs->adapt_comm_num_segments % cs->adapt_comm_num_in_use_flags);
    }
    cs->adapt_segs_per_inuse_flag = 
        cs->adapt_comm_num_segments / cs->adapt_comm_num_in_use_flags;

    if (cs->adapt_tree_degree > cs->adapt_control_size) {
        opal_show_help("help-mpi-coll-adapt-cuda.txt", 
                       "tree-degree-larger-than-control", true,
                       cs->adapt_tree_degree, cs->adapt_control_size);
        cs->adapt_tree_degree = cs->adapt_control_size;
    }
    if (cs->adapt_tree_degree > 255) {
        opal_show_help("help-mpi-coll-adapt-cuda.txt", 
                       "tree-degree-larger-than-255", true,
                       cs->adapt_tree_degree);
        cs->adapt_tree_degree = 255;
    }

    coll_adapt_cuda_shared_mem_used_data = (int)(4 * cs->adapt_control_size +
        (cs->adapt_comm_num_in_use_flags * cs->adapt_control_size) +
        (cs->adapt_comm_num_segments * (cs->adapt_info_comm_size * cs->adapt_control_size * 2)) +
        (cs->adapt_comm_num_segments * (cs->adapt_info_comm_size * cs->adapt_fragment_size)));

    return OMPI_SUCCESS;
}

/*
 * Register MCA params
 */
static int adapt_cuda_register(void)
{
    mca_base_component_t *c = &mca_coll_adapt_cuda_component.super.collm_version;
    mca_coll_adapt_cuda_component_t *cs = &mca_coll_adapt_cuda_component;

    /* If we want to be selected (i.e., all procs on one node), then
       we should have a high priority */

    cs->adapt_cuda_priority = 0;
    (void) mca_base_component_var_register(c, "priority", "Priority of the sm coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_cuda_priority);

    cs->adapt_control_size = 4096;
    (void) mca_base_component_var_register(c, "control_size",
                                           "Length of the control data -- should usually be either the length of a cache line on most SMPs, or the size of a page on machines that support direct memory affinity page placement (in bytes)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_control_size);

    cs->adapt_fragment_size = 8192;
    (void) mca_base_component_var_register(c, "fragment_size",
                                           "Fragment size (in bytes) used for passing data through shared memory (will be rounded up to the nearest control_size size)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_fragment_size);

    cs->adapt_comm_num_in_use_flags = 2;
    (void) mca_base_component_var_register(c, "comm_in_use_flags",
                                           "Number of \"in use\" flags, used to mark a message passing area segment as currently being used or not (must be >= 2 and <= comm_num_segments)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_comm_num_in_use_flags);

    cs->adapt_comm_num_segments = 8;
    (void) mca_base_component_var_register(c, "comm_num_segments",
                                           "Number of segments in each communicator's shared memory message passing area (must be >= 2, and must be a multiple of comm_in_use_flags)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_comm_num_segments);

    cs->adapt_tree_degree = 4;
    (void) mca_base_component_var_register(c, "tree_degree",
                                           "Degree of the tree for tree-based operations (must be => 1 and <= min(control_size, 255))",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_tree_degree);

    /* INFO: Calculate how much space we need in the per-communicator
       shmem data segment.  This formula taken directly from
       coll_adapt_cuda_module.c. */
    cs->adapt_info_comm_size = 4;
    (void) mca_base_component_var_register(c, "info_num_procs",
                                           "Number of processes to use for the calculation of the shared_mem_size MCA information parameter (must be => 2)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_info_comm_size);

    coll_adapt_cuda_shared_mem_used_data = (int)(4 * cs->adapt_control_size +
        (cs->adapt_comm_num_in_use_flags * cs->adapt_control_size) +
        (cs->adapt_comm_num_segments * (cs->adapt_info_comm_size * cs->adapt_control_size * 2)) +
        (cs->adapt_comm_num_segments * (cs->adapt_info_comm_size * cs->adapt_fragment_size)));

    (void) mca_base_component_var_register(c, "shared_mem_used_data",
                                           "Amount of shared memory used, per communicator, in the shared memory data area for info_num_procs processes (in bytes)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &coll_adapt_cuda_shared_mem_used_data);
                                           
    /* init a mpool for pined cpu buffer */
    cs->pined_cpu_mpool = coll_adapt_cuda_mpool_create();

    return adapt_cuda_verify_mca_variables();
}

static int coll_adapt_cuda_progress()
{
    char *context;
    while (1 == progress_one_cuda_op_event((void **)&context)) {
        if (context != NULL) {
            int *flag = (int *)(context + sizeof(opal_free_list_item_t));
            if (*flag == COLL_ADAPT_CUDA_CONTEXT_FLAGS_REDUCE) {
              //  opal_output(0, "reduce call back\n");
                mca_coll_adapt_cuda_reduce_context_t *reduce_context = (mca_coll_adapt_cuda_reduce_context_t *)context;
                assert(reduce_context->cuda_callback != NULL);
                reduce_context->cuda_callback(reduce_context);
            }
        }
    }
    while (1 == progress_one_cuda_memcpy_event((void **)&context)) {
        if (context != NULL) {
            int *flag = (int *)(context + sizeof(opal_free_list_item_t));
            if (*flag == COLL_ADAPT_CUDA_CONTEXT_FLAGS_BCAST) {
                opal_output(0, "bcast call back\n");
                mca_coll_adapt_cuda_bcast_context_t *bcast_context = (mca_coll_adapt_cuda_bcast_context_t *)context;
                assert(bcast_context->cuda_callback != NULL);
                bcast_context->cuda_callback(bcast_context);
            } else if (*flag == COLL_ADAPT_CUDA_CONTEXT_FLAGS_REDUCE) {
              //  opal_output(0, "reduce call back\n");
                mca_coll_adapt_cuda_reduce_context_t *reduce_context = (mca_coll_adapt_cuda_reduce_context_t *)context;
                assert(reduce_context->cuda_callback != NULL);
                reduce_context->cuda_callback(reduce_context);
            }
        }
    }
    return OMPI_SUCCESS;
}