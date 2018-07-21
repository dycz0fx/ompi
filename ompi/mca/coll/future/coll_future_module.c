/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2009-2013 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2012 Los Alamos National Security, LLC.
 *                         All rights reserved.
 * Copyright (c) 2014-2015 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
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
 * Warning: this is not for the faint of heart -- don't even bother
 * reading this source code if you don't have a strong understanding
 * of nested data structures and pointer math (remember that
 * associativity and order of C operations is *critical* in terms of
 * pointer math!).
 */

#include "ompi_config.h"

#include <stdio.h>
#include <string.h>
#ifdef HAVE_SCHED_H
#include <sched.h>
#endif
#include <sys/types.h>
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif  /* HAVE_SYS_MMAN_H */
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */

#include "mpi.h"
#include "opal_stdint.h"
#include "opal/mca/hwloc/base/base.h"
#include "opal/util/os_path.h"

#include "ompi/communicator/communicator.h"
#include "ompi/group/group.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/rte/rte.h"
#include "ompi/proc/proc.h"
#include "coll_future.h"

#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"


/*
 * Local functions
 */
static int future_module_enable(mca_coll_base_module_t *module,
                                struct ompi_communicator_t *comm);
static int mca_coll_future_module_disable(mca_coll_base_module_t *module,
                                          struct ompi_communicator_t *comm);

/*
 * Module constructor
 */
static void mca_coll_future_module_construct(mca_coll_future_module_t *module)
{
    module->enabled = false;
    module->super.coll_module_disable = mca_coll_future_module_disable;
    module->cached_comm = NULL;
    module->cached_sm_comm = NULL;
    module->cached_leader_comm = NULL;
    module->cached_vranks = NULL;
}

/*
 * Module destructor
 */
static void mca_coll_future_module_destruct(mca_coll_future_module_t *module)
{
    module->enabled = false;
    if (module->cached_sm_comm != NULL) {
        ompi_comm_free(&(module->cached_sm_comm));
    }
    if (module->cached_leader_comm != NULL) {
        ompi_comm_free(&(module->cached_leader_comm));
    }
    if (module->cached_vranks != NULL) {
        free(module->cached_vranks);
    }
}

/*
 * Module disable
 */
static int mca_coll_future_module_disable(mca_coll_base_module_t *module, struct ompi_communicator_t *comm)
{
    return OMPI_SUCCESS;
}


OBJ_CLASS_INSTANCE(mca_coll_future_module_t,
                   mca_coll_base_module_t,
                   mca_coll_future_module_construct,
                   mca_coll_future_module_destruct);

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.  This function is invoked exactly
 * once.
 */
int mca_coll_future_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads)
{
    /* if no session directory was created, then we cannot be used */
    if (NULL == ompi_process_info.job_session_dir) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    /* Don't do much here because we don't really want to allocate any
     future memory until this component is selected to be used. */
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:future:init_query: pick me! pick me!");
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_future_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    mca_coll_future_module_t *future_module;
    
    /* If we're intercomm, or if there's only one process in the
     communicator */
    if (OMPI_COMM_IS_INTER(comm) || 1 == ompi_comm_size(comm)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:future:comm_query (%d/%s): intercomm, comm is too small, or not all peers local; disqualifying myself", comm->c_contextid, comm->c_name);
        return NULL;
    }
    
    /* Get the priority level attached to this module. If priority is less
     * than or equal to 0, then the module is unavailable. */
    *priority = mca_coll_future_component.future_priority;
    if (mca_coll_future_component.future_priority <= 0) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:future:comm_query (%d/%s): priority too low; disqualifying myself", comm->c_contextid, comm->c_name);
        return NULL;
    }
    
    future_module = OBJ_NEW(mca_coll_future_module_t);
    if (NULL == future_module) {
        return NULL;
    }
    
    /* All is good -- return a module */
    future_module->super.coll_module_enable = future_module_enable;
    future_module->super.ft_event        = NULL;
    future_module->super.coll_allgather  = NULL;
    future_module->super.coll_allgatherv = NULL;
    future_module->super.coll_allreduce  = NULL;
    future_module->super.coll_alltoall   = NULL;
    future_module->super.coll_alltoallv  = NULL;
    future_module->super.coll_alltoallw  = NULL;
    future_module->super.coll_barrier    = NULL;
    future_module->super.coll_bcast      = mca_coll_future_bcast_intra;
    future_module->super.coll_exscan     = NULL;
    future_module->super.coll_gather     = NULL;
    future_module->super.coll_gatherv    = NULL;
    future_module->super.coll_reduce     = NULL;
    future_module->super.coll_reduce_scatter = NULL;
    future_module->super.coll_scan       = NULL;
    future_module->super.coll_scatter    = NULL;
    future_module->super.coll_scatterv   = NULL;
    
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:future:comm_query (%d/%s): pick me! pick me!",
                        comm->c_contextid, comm->c_name);
    return &(future_module->super);
}


/*
 * Init module on the communicator
 */
static int future_module_enable(mca_coll_base_module_t *module,
                                struct ompi_communicator_t *comm)
{
    return OMPI_SUCCESS;
}

int ompi_coll_future_lazy_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm)
{
    return OMPI_SUCCESS;
}
