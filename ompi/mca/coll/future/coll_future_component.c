/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2009 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2011-2015 Los Alamos National Security, LLC.
 *                         All rights reserved.
 * Copyright (c) 2015      Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 *
 * Most of the description of the data layout is in the
 * coll_future_module.c file.
 */

#include "ompi_config.h"

#include "opal/util/show_help.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "coll_future.h"

/*
 * Public string showing the coll ompi_future component version number
 */
const char *mca_coll_future_component_version_string =
"Open MPI future collective MCA component version " OMPI_VERSION;


/*
 * Local functions
 */
static int future_open(void);
static int future_close(void);
static int future_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_future_component_t mca_coll_future_component = {
    
    /* First, fill in the super */
    
    {
        /* First, the mca_component_t struct containing meta
         information about the component itself */
        
        .collm_version = {
            MCA_COLL_BASE_VERSION_2_0_0,
            
            /* Component name and version */
            .mca_component_name = "future",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),
            
            /* Component functions */
            .mca_open_component = future_open,
            .mca_close_component = future_close,
            .mca_register_component_params = future_register,
        },
        .collm_data = {
            /* The component is not checkpoint ready */
            MCA_BASE_METADATA_PARAM_NONE
        },
        
        /* Initialization / querying functions */
        
        .collm_init_query = mca_coll_future_init_query,
        .collm_comm_query = mca_coll_future_comm_query,
    },
    
    /* future-component specifc information */
    
    /* (default) priority */
    20,
};

/*
 * Init the component
 */
static int future_open(void){
    mca_coll_future_component_t *cs = &mca_coll_future_component;
    if (cs->future_auto_tune) {
        cs->future_auto_tuned = (selection *)malloc(2*cs->future_auto_tune_n * cs->future_auto_tune_c * cs->future_auto_tune_m *  sizeof(selection));
        char *filename = "/lustre/project/k1205/lei/xi/results/auto2/auto_tuned.bin";
        FILE *file = fopen(filename, "r");
        fread(cs->future_auto_tuned, sizeof(selection), 2*cs->future_auto_tune_n * cs->future_auto_tune_c * cs->future_auto_tune_m, file);
        fclose(file);
    }
    return OMPI_SUCCESS;
}


/*
 * Shut down the component
 */
static int future_close(void)
{
    mca_coll_future_component_t *cs = &mca_coll_future_component;
    if (cs->future_auto_tune && cs->future_auto_tuned != NULL){
        free(cs->future_auto_tuned);
        cs->future_auto_tuned = NULL;
    }
    return OMPI_SUCCESS;
}


/*
 * Register MCA params
 */
static int future_register(void)
{
    mca_base_component_t *c = &mca_coll_future_component.super.collm_version;
    mca_coll_future_component_t *cs = &mca_coll_future_component;
    
    cs->future_priority = 0;
    (void) mca_base_component_var_register(c, "priority", "Priority of the future coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_priority);
    
    int coll_future_verbose = 0;
    (void) mca_base_component_var_register(c, "verbose",
                                           "Verbose level",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &coll_future_verbose);
    cs->future_output = opal_output_open(NULL);
    opal_output_set_verbosity(cs->future_output, coll_future_verbose);
    
    cs->future_bcast_up_segsize = 65536;
    (void) mca_base_component_var_register(c, "bcast_up_segsize",
                                           "up level segment size for bcast",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_bcast_up_segsize);
    
    cs->future_bcast_low_segsize = 524288;
    (void) mca_base_component_var_register(c, "bcast_low_segsize",
                                           "low level segment size for bcast",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_bcast_low_segsize);
    
    cs->future_bcast_up_module = 0;
    (void) mca_base_component_var_register(c, "bcast_up_module",
                                           "up level module for bcast, 0 libnbc, 1 adapt",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_bcast_up_module);
    
    cs->future_bcast_low_module = 0;
    (void) mca_base_component_var_register(c, "bcast_low_module",
                                           "low level module for bcast, 0 sm, 1 shared",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_bcast_low_module);
    
    cs->future_allreduce_segsize = 524288;
    (void) mca_base_component_var_register(c, "allreduce_segsize",
                                           "segment size for allreduce",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_allreduce_segsize);

    cs->future_allreduce_up_module = 0;
    (void) mca_base_component_var_register(c, "allreduce_up_module",
                                           "up level module for allreduce, 0 libnbc, 1 adapt",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_allreduce_up_module);
    
    cs->future_allreduce_low_module = 0;
    (void) mca_base_component_var_register(c, "allreduce_low_module",
                                           "low level module for allreduce, 0 sm, 1 shared",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_allreduce_low_module);
    
    cs->future_allgather_up_module = 0;
    (void) mca_base_component_var_register(c, "allgather_up_module",
                                           "up level module for allgather, 0 libnbc, 1 adapt",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_allgather_up_module);
    
    cs->future_allgather_low_module = 0;
    (void) mca_base_component_var_register(c, "allgather_low_module",
                                           "low level module for allgather, 0 sm, 1 shared",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_allgather_low_module);
    
    cs->future_gather_up_module = 0;
    (void) mca_base_component_var_register(c, "gather_up_module",
                                           "up level module for gather, 0 libnbc, 1 adapt",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_gather_up_module);
    
    cs->future_gather_low_module = 0;
    (void) mca_base_component_var_register(c, "gather_low_module",
                                           "low level module for gather, 0 sm, 1 shared",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_gather_low_module);
    
    cs->future_scatter_up_module = 0;
    (void) mca_base_component_var_register(c, "scatter_up_module",
                                           "up level module for scatter, 0 libnbc, 1 adapt",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_scatter_up_module);
    
    cs->future_scatter_low_module = 0;
    (void) mca_base_component_var_register(c, "scatter_low_module",
                                           "low level module for scatter, 0 sm, 1 shared",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_scatter_low_module);
    
    cs->future_auto_tune = 0;
    (void) mca_base_component_var_register(c, "auto_tune",
                                           "whether enable auto tune, 0 disable, 1 enable, default 0",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_auto_tune);

    cs->future_auto_tune_n = 4;
    (void) mca_base_component_var_register(c, "auto_tune_n",
                                           "auto tune n",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_auto_tune_n);
    
    cs->future_auto_tune_c = 1;
    (void) mca_base_component_var_register(c, "auto_tune_c",
                                           "auto tune c",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_auto_tune_c);

    cs->future_auto_tune_m = 26;
    (void) mca_base_component_var_register(c, "auto_tune_m",
                                           "auto tune n",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->future_auto_tune_m);

    return OMPI_SUCCESS;
}

