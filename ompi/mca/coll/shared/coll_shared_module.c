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
#include "coll_shared.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"


/*
 * Local functions
 */
static int shared_module_enable(mca_coll_base_module_t *module,
                          struct ompi_communicator_t *comm);
static int mca_coll_shared_module_disable(mca_coll_base_module_t *module,
                          struct ompi_communicator_t *comm);

/*
 * Module constructor
 */
static void mca_coll_shared_module_construct(mca_coll_shared_module_t *module)
{
    module->enabled = false;
    module->sm_data_ptr = NULL;
    module->sm_data_win = NULL;
    //module->data_buf = NULL;
    module->sm_ctrl_ptr = NULL;
    module->sm_ctrl_win = NULL;
    //module->ctrl_buf = NULL;
    module->super.coll_module_disable = mca_coll_shared_module_disable;
}

/*
 * Module destructor
 */
static void mca_coll_shared_module_destruct(mca_coll_shared_module_t *module)
{
    return;
}

/*
 * Module disable
 */
static int mca_coll_shared_module_disable(mca_coll_base_module_t *module, struct ompi_communicator_t *comm)
{
    mca_coll_shared_module_t *m = (mca_coll_shared_module_t *)module;
    m->enabled = false;
    /* windows will be free at ompi_mpi_finalize.c:324 */
    /*
    if (m->sm_data_win != NULL) {
        ompi_win_free(m->sm_data_win);
    }
    if (m->sm_ctrl_win != NULL) {
        ompi_win_free(m->sm_ctrl_win);
    }
     */
    /*
    if (m->data_buf != NULL) {
        free(m->data_buf);
    }
    if (m->ctrl_buf != NULL) {
        free(m->ctrl_buf);
    }
    */
    return OMPI_SUCCESS;
}


OBJ_CLASS_INSTANCE(mca_coll_shared_module_t,
                   mca_coll_base_module_t,
                   mca_coll_shared_module_construct,
                   mca_coll_shared_module_destruct);

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.  This function is invoked exactly
 * once.
 */
int mca_coll_shared_init_query(bool enable_progress_threads,
                           bool enable_mpi_threads)
{
    /* if no session directory was created, then we cannot be used */
    if (NULL == ompi_process_info.job_session_dir) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    /* Don't do much here because we don't really want to allocate any
       shared memory until this component is selected to be used. */
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:shared:init_query: pick me! pick me!");
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_shared_comm_query(struct ompi_communicator_t *comm, int *priority)
{
    mca_coll_shared_module_t *shared_module;

    /* If we're intercomm, or if there's only one process in the
     communicator, or if not all the processes in the communicator
     are not on this node, then we don't want to run */
    if (OMPI_COMM_IS_INTER(comm) || 1 == ompi_comm_size(comm) || ompi_group_have_remote_peers (comm->c_local_group)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:shared:comm_query (%d/%s): intercomm, comm is too small, or not all peers local; disqualifying myself", comm->c_contextid, comm->c_name);
	return NULL;
    }

    /* Get the priority level attached to this module. If priority is less
     * than or equal to 0, then the module is unavailable. */
    *priority = mca_coll_shared_component.shared_priority;
    if (mca_coll_shared_component.shared_priority <= 0) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:shared:comm_query (%d/%s): priority too low; disqualifying myself", comm->c_contextid, comm->c_name);
	return NULL;
    }

    shared_module = OBJ_NEW(mca_coll_shared_module_t);
    if (NULL == shared_module) {
        return NULL;
    }

    /* All is good -- return a module */
    shared_module->super.coll_module_enable = shared_module_enable;
    shared_module->super.ft_event        = NULL;
    shared_module->super.coll_allgather  = NULL;
    shared_module->super.coll_allgatherv = NULL;
    shared_module->super.coll_allreduce  = mca_coll_shared_allreduce_intra; //mca_coll_shared_allreduce_intra
    shared_module->super.coll_alltoall   = NULL;
    shared_module->super.coll_alltoallv  = NULL;
    shared_module->super.coll_alltoallw  = NULL;
    shared_module->super.coll_barrier    = NULL; //mca_coll_shared_barrier_intra;
    shared_module->super.coll_bcast      = mca_coll_shared_bcast_intra; //mca_coll_shared_bcast_linear_intra mca_coll_shared_bcast_linear_nofence_intra mca_coll_shared_bcast_binary
    shared_module->super.coll_exscan     = NULL;
    shared_module->super.coll_gather     = NULL;
    shared_module->super.coll_gatherv    = NULL;
    shared_module->super.coll_reduce     = mca_coll_shared_reduce_intra; //mca_coll_shared_reduce_intra; //mca_coll_shared_reduce_intra;
    shared_module->super.coll_reduce_scatter = NULL;
    shared_module->super.coll_scan       = NULL;
    shared_module->super.coll_scatter    = NULL;
    shared_module->super.coll_scatterv   = NULL;

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:shared:comm_query (%d/%s): pick me! pick me!",
                        comm->c_contextid, comm->c_name);
    return &(shared_module->super);
}


/*
 * Init module on the communicator
 */
static int shared_module_enable(mca_coll_base_module_t *module,
                            struct ompi_communicator_t *comm)
{
    return OMPI_SUCCESS;
}

int ompi_coll_shared_lazy_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm)
{
    //printf("shared_module_lazy_enable start\n");
    int i;
    //comm->c_coll->coll_allreduce = ompi_coll_base_allreduce_intra_recursivedoubling;
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
 
    
    int var_id;
    int tmp_priority = 100;
    const int *origin_priority = NULL;
    int tmp_origin = 0;
    //const int *tmp = NULL;
    mca_base_var_find_by_name("coll_tuned_priority", &var_id);
    mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
    tmp_origin = *origin_priority;
    mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
    mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
    
    int size = ompi_comm_size(comm);
    
    /* create a shared memory to store data for every process */
    int max_seg_size = MAX_SEG_SIZE;
    ompi_win_allocate_shared(max_seg_size*sizeof(char), sizeof(char), (opal_info_t *)(&ompi_mpi_info_null), comm, &shared_module->sm_data_ptr, &shared_module->sm_data_win);
    size_t data_size[size];
    int data_disp[size];
    //shared_module->data_buf = (char **)malloc(sizeof(char *) * size);
    /* get data shared memory */
    for (i=0; i<size; i++) {
        shared_module->sm_data_win->w_osc_module->osc_win_shared_query(shared_module->sm_data_win, i, &(data_size[i]), &(data_disp[i]), &(shared_module->data_buf[i]));
    }
    /* create a shared memory to store control message on every node */
    ompi_win_allocate_shared(1*sizeof(int), sizeof(int), (opal_info_t *)(&ompi_mpi_info_null), comm, &shared_module->sm_ctrl_ptr, &shared_module->sm_ctrl_win);
    size_t ctrl_size[size];
    int ctrl_disp[size];
    //shared_module->ctrl_buf = (int **)malloc(sizeof(int *) * size);
    /* get ctrl shared memory */
    for (i=0; i<size; i++) {
        shared_module->sm_ctrl_win->w_osc_module->osc_win_shared_query(shared_module->sm_ctrl_win, i, &(ctrl_size[i]), &(ctrl_disp[i]), &(shared_module->ctrl_buf[i]));
    }

    shared_module->enabled = true;
    //comm->c_coll->coll_allreduce = mca_coll_shared_allreduce_intra;
    
    mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
     
    //printf("shared_module_lazy_enable end\n");
    return OMPI_SUCCESS;
}
