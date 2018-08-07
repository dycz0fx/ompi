/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2009 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/** @file */

#ifndef MCA_COLL_SM_EXPORT_H
#define MCA_COLL_SM_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
//#include "opal/datatype/opal_convertor.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/communicator/communicator.h"
#include "ompi/include/mpi.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/info.h"
#include "ompi/op/op.h"
#include "opal/runtime/opal_progress.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_tags.h"

BEGIN_C_DECLS

/**
 * Structure to hold the adapt coll component.  First it holds the
 * base coll component, and then holds a bunch of
 * adapt-coll-component-specific stuff (e.g., current MCA param
 * values).
 */
typedef struct mca_coll_adapt_component_t {
    /** Base coll component */
    mca_coll_base_component_2_0_0_t super;
    
    /** MCA parameter: Priority of this component */
    int adapt_priority;
    /* whether output the log message */
    int adapt_output;
    int adapt_context_free_list_min;
    int adapt_context_free_list_max;
    int adapt_context_free_list_inc;
} mca_coll_adapt_component_t;

/** Coll adapt module */
typedef struct mca_coll_adapt_module_t {
    /** Base module */
    mca_coll_base_module_t super;
    
    /* Whether this module has been lazily initialized or not yet */
    bool enabled;
    
} mca_coll_adapt_module_t;
OBJ_CLASS_DECLARATION(mca_coll_adapt_module_t);

/**
 * Global component instance
 */
OMPI_MODULE_DECLSPEC extern mca_coll_adapt_component_t mca_coll_adapt_component;

/*
 * coll module functions
 */
int mca_coll_adapt_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads);

mca_coll_base_module_t *
mca_coll_adapt_comm_query(struct ompi_communicator_t *comm, int *priority);

/* use to select algorithm */
typedef struct mca_coll_adapt_algorithm_index_s {
    int algorithm_index;
    uintptr_t algorithm_fn_ptr;
}mca_coll_adapt_algorithm_index_t;

/* Lazily enable a module (since it involves expensive/slow mmap
 allocation, etc.) */
int ompi_coll_adapt_lazy_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm);

int mca_coll_adapt_ibcast_init(void);
int mca_coll_adapt_ibcast_fini(void);
int mca_coll_adapt_ibcast_intra(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module);
int mca_coll_adapt_ibcast_tuned(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag);
int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, int ibcast_tag);

/* print tree for test */
static inline void print_tree(ompi_coll_tree_t* tree, int rank) {
    int i;
    printf("[%d, prev = %d, next_size = %d, root =%d]:", rank, tree->tree_prev, tree->tree_nextsize, tree->tree_root);
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
    OPAL_THREAD_LOCK ((*request)->req_lock);
    (*request)->req_state = OMPI_REQUEST_INVALID;
    OPAL_THREAD_UNLOCK ((*request)->req_lock);
    OBJ_RELEASE(*request);
    *request = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

END_C_DECLS

#endif /* MCA_COLL_SM_EXPORT_H */
