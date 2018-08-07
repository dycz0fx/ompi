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
 * coll_adapt_module.c file.
 */

#include "ompi_config.h"

#include "opal/util/show_help.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "coll_adapt.h"


/*
 * Public string showing the coll ompi_adapt component version number
 */
const char *mca_coll_adapt_component_version_string =
"Open MPI adapt collective MCA component version " OMPI_VERSION;


/*
 * Local functions
 */
static int adapt_close(void);
static int adapt_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_adapt_component_t mca_coll_adapt_component = {
    
    /* First, fill in the super */
    
    {
        /* First, the mca_component_t struct containing meta
         information about the component itself */
        
        .collm_version = {
            MCA_COLL_BASE_VERSION_2_0_0,
            
            /* Component name and version */
            .mca_component_name = "adapt",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),
            
            /* Component functions */
            .mca_close_component = adapt_close,
            .mca_register_component_params = adapt_register,
        },
        .collm_data = {
            /* The component is not checkpoint ready */
            MCA_BASE_METADATA_PARAM_NONE
        },
        
        /* Initialization / querying functions */
        
        .collm_init_query = mca_coll_adapt_init_query,
        .collm_comm_query = mca_coll_adapt_comm_query,
    },
    
    /* adapt-component specifc information */
    
    /* (default) priority */
    /* JMS temporarily lowered until we can get more testing */
    0,
};


/*
 * Shut down the component
 */
static int adapt_close(void)
{
    mca_coll_adapt_ibcast_fini();
    return OMPI_SUCCESS;
}


/*
 * Register MCA params
 */
static int adapt_register(void)
{
    mca_base_component_t *c = &mca_coll_adapt_component.super.collm_version;
    mca_coll_adapt_component_t *cs = &mca_coll_adapt_component;
    
    /* If we want to be selected (i.e., all procs on one node), then
     we should have a high priority */
    
    cs->adapt_priority = 0;
    
    (void) mca_base_component_var_register(c, "priority", "Priority of the adapt coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_priority);
    
    int coll_adapt_verbose = 0;
    (void) mca_base_component_var_register(c, "verbose",
                                           "Verbose level",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &coll_adapt_verbose);
    cs->adapt_output = opal_output_open(NULL);
    opal_output_set_verbosity(cs->adapt_output, coll_adapt_verbose);
    
    cs->adapt_context_free_list_min = 10;
    (void) mca_base_component_var_register(c, "context_free_list_max",
                                           "Minimum number of segment in context free list",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_context_free_list_min);
    
    cs->adapt_context_free_list_max = 10000;
    (void) mca_base_component_var_register(c, "context_free_list_min",
                                           "Maximum number of segment in context free list",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_context_free_list_max);
    
    cs->adapt_context_free_list_inc = 10;
    (void) mca_base_component_var_register(c, "context_free_list_inc",
                                           "Increasement number of segment in context free list",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &cs->adapt_context_free_list_inc);

    mca_coll_adapt_ibcast_init();
    return OMPI_SUCCESS;
}