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
#include <math.h>
#include <limits.h>


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
    module->cached_low_comms = NULL;
    module->cached_up_comms = NULL;
    module->cached_vranks = NULL;
    module->cached_topo = NULL;
    module->is_mapbycore = false;
}

/*
 * Module destructor
 */
static void mca_coll_future_module_destruct(mca_coll_future_module_t *module)
{
    module->enabled = false;
    if (module->cached_low_comms != NULL) {
        ompi_comm_free(&(module->cached_low_comms[0]));
        ompi_comm_free(&(module->cached_low_comms[1]));
        module->cached_low_comms[0] = NULL;
        module->cached_low_comms[1] = NULL;
        free(module->cached_low_comms);
        module->cached_low_comms = NULL;
    }
    if (module->cached_up_comms != NULL) {
        ompi_comm_free(&(module->cached_up_comms[0]));
        ompi_comm_free(&(module->cached_up_comms[1]));
        module->cached_up_comms[0] = NULL;
        module->cached_up_comms[1] = NULL;
        free(module->cached_up_comms);
        module->cached_up_comms = NULL;
    }
    if (module->cached_vranks != NULL) {
        free(module->cached_vranks);
        module->cached_vranks = NULL;
    }
    if (module->cached_topo != NULL) {
        free(module->cached_topo);
        module->cached_topo = NULL;
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
    if (OMPI_COMM_IS_INTER(comm) || 1 == ompi_comm_size(comm) || !ompi_group_have_remote_peers (comm->c_local_group)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:future:comm_query (%d/%s): intercomm, comm is too small, only on one node; disqualifying myself", comm->c_contextid, comm->c_name);
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
    future_module->super.coll_allgather  = mca_coll_future_allgather_intra;
    future_module->super.coll_allgatherv = NULL;
    future_module->super.coll_allreduce  = mca_coll_future_allreduce_intra_sync;
    future_module->super.coll_alltoall   = NULL;
    future_module->super.coll_alltoallv  = NULL;
    future_module->super.coll_alltoallw  = NULL;
    future_module->super.coll_barrier    = NULL;
    future_module->super.coll_bcast      = mca_coll_future_bcast_intra_sync;
    future_module->super.coll_exscan     = NULL;
    future_module->super.coll_gather     = ompi_coll_future_gather_intra;
    future_module->super.coll_gatherv    = NULL;
    future_module->super.coll_reduce     = NULL;
    future_module->super.coll_reduce_scatter = NULL;
    future_module->super.coll_scan       = NULL;
    future_module->super.coll_scatter    = ompi_coll_future_scatter_intra;
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

int future_request_free(ompi_request_t** request)
{
    (*request)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*request);
    *request = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

void mca_coll_future_comm_create(struct ompi_communicator_t *comm, mca_coll_future_module_t *future_module){
    /* use cached communicators if possible */
    if (future_module->cached_comm == comm && future_module->cached_low_comms != NULL && future_module->cached_up_comms != NULL && future_module->cached_vranks != NULL) {
        return;
    }
    /* create communicators if there is no cached communicator */
    else {
        int low_rank, low_size;
        int up_rank;
        int w_rank = ompi_comm_rank(comm);
        int w_size = ompi_comm_size(comm);
        ompi_communicator_t **low_comms = (struct ompi_communicator_t **)malloc(sizeof(struct ompi_communicator_t *) * 2);
        ompi_communicator_t **up_comms = (struct ompi_communicator_t **)malloc(sizeof(struct ompi_communicator_t *) * 2);
        /* create low_comms which contain all the process on a node */
        const int *origin_priority = NULL;
        //const int *tmp = NULL;
        /* lower future module priority */
        int future_var_id;
        int tmp_future_priority = 0;
        int tmp_future_origin = 0;
        mca_base_var_find_by_name("coll_future_priority", &future_var_id);
        mca_base_var_get_value(future_var_id, &origin_priority, NULL, NULL);
        tmp_future_origin = *origin_priority;
        mca_base_var_set_flag(future_var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(future_var_id, &tmp_future_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        comm->c_coll->coll_allreduce = ompi_coll_base_allreduce_intra_recursivedoubling;
        comm->c_coll->coll_allgather = ompi_coll_base_allgather_intra_recursivedoubling;
        
        int var_id;
        int tmp_priority = 100;
        int tmp_origin = 0;
        /* set up low_comms[0] with sm module */
        mca_base_var_find_by_name("coll_sm_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] sm_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("sm_priority after set %d %d\n", *tmp);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &(low_comms[0]));
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] sm_priority set back %d\n", w_rank, *tmp);
        low_size = ompi_comm_size(low_comms[0]);
        low_rank = ompi_comm_rank(low_comms[0]);
        
        /* set up low_comms[1] with shared module */
        mca_base_var_find_by_name("coll_shared_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] shared_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("shared_priority after set %d %d\n", *tmp);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &(low_comms[1]));
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);

        /* create up_comms[0] with libnbc which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_libnbc_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] libnbc_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("libnbc_priority after set %d %d\n", *tmp);
        ompi_comm_split(comm, low_rank, w_rank, &(up_comms[0]), false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] libnbc_priority set back %d\n", w_rank, *tmp);
        up_rank = ompi_comm_rank(up_comms[0]);
        
        /* create up_comms[1] with adapt which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_adapt_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] adapt_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("adapt_priority after set %d %d\n", *tmp);
        ompi_comm_split(comm, low_rank, w_rank, &(up_comms[1]), false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] adapt_priority set back %d\n", w_rank, *tmp);
        
        int *vranks = malloc(sizeof(int) * w_size);
        /* do allgather to gather vrank from each process so every process will know other processes vrank*/
        int vrank = low_size * up_rank + low_rank;
        comm->c_coll->coll_allgather(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm, comm->c_coll->coll_allgather_module);
        future_module->cached_comm = comm;
        future_module->cached_low_comms = low_comms;
        future_module->cached_up_comms = up_comms;
        future_module->cached_vranks = vranks;
        
        mca_base_var_set_value(future_var_id, &tmp_future_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        comm->c_coll->coll_allreduce = mca_coll_future_allreduce_intra_sync;
        comm->c_coll->coll_allgather = mca_coll_future_allgather_intra;
    }
}

int mca_coll_future_pow10_int(int pow_value){
    int i, result = 1;
    for (i=0; i<pow_value; i++) {
        result *= 10;
    }
    return result;
}

int mca_coll_future_hostname_to_number(char* hostname, int size) {
    int i=0, j=0;
    char * number_array = (char *)malloc(sizeof(char)*size);
    while (hostname[i] != '\0'){
        if(hostname[i] >= '0' && hostname[i] <= '9'){
            number_array[j++] = hostname[i];
        }
        i++;
    }
    int number = 0;
    for (i=0; i<j; i++){
        number += (number_array[i]-'0') * mca_coll_future_pow10_int(j-1-i);
    }
    free(number_array);
    return number;
}

void mca_coll_future_topo_get(int *topo, struct ompi_communicator_t* comm, int num_topo_level){
    int * self_topo = (int *)malloc(sizeof(int) * num_topo_level);
    /* set daemon vpid */
    //self_topo[0] = OMPI_RTE_MY_NODEID;
    char hostname[1024];
    //printf("[%d]: %s\n", ompi_comm_rank(comm), hostname);
    gethostname(hostname, 1024);
    self_topo[0] = mca_coll_future_hostname_to_number(hostname, 1024);
    //set core id
    self_topo[1] = ompi_comm_rank(comm);
    
    //printf("[topo %d]: %d %d\n", ompi_comm_rank(comm), self_topo[0], self_topo[1]);
    //fflush(stdout);
    //do allgather
    ompi_coll_base_allgather_intra_bruck(self_topo, num_topo_level, MPI_INT, topo, num_topo_level, MPI_INT, comm, comm->c_coll->coll_allgather_module);
    //printf("[topo %d]: after allgather\n", ompi_comm_rank(comm));
    //fflush(stdout);
    free(self_topo);
    return;
}

void mca_coll_future_topo_sort(int *topo, int start, int end, int size, int level, int num_topo_level){
    if (level > num_topo_level-1 || start >= end) {
        return;
    }
    int i, j;
    //    printf("Sort [start %d, end %d, level %d]\n", start, end, level);
    //    printf("Before sort: ");
    //    for (i=start; i<=end; i++) {
    //        printf("%d ", topo[i*num_topo_level+level]);
    //    }
    //    printf("\n");
    int min = INT_MAX;
    int min_loc = -1;
    for (i=start; i<=end; i++) {
        //find min
        for (j=i; j<=end; j++) {
            if (topo[j*num_topo_level+level] < min) {
                min = topo[j*num_topo_level+level];
                min_loc = j;
                //printf("i %d j %d min %d, min_loc %d\n", i, j, min, min_loc);
                
            }
        }
        //swap i and min_loc
        //printf("min %d, min_loc %d\n", min, min_loc);
        int temp;
        for (j=0; j<num_topo_level; j++) {
            temp = topo[i*num_topo_level+j];
            topo[i*num_topo_level+j] = topo[min_loc*num_topo_level+j];
            topo[min_loc*num_topo_level+j] = temp;
        }
        min = INT_MAX;
        min_loc = -1;
    }
    //    printf("After sort: ");
    //    for (i=start; i<=end; i++) {
    //        printf("%d ", topo[i*num_topo_level+level]);
    //    }
    //    printf("\n");
    int last = 0;
    int new_start = 0;
    int new_end = 0;
    for (i=start; i<=end; i++) {
        if (i == start) {
            last = topo[i*num_topo_level+level];
            new_start = start;
        }
        else if (i == end) {
            new_end = end;
            mca_coll_future_topo_sort(topo, new_start, new_end, size, level+1, num_topo_level);
        }
        else if (last != topo[i*num_topo_level+level]) {
            new_end = i-1;
            mca_coll_future_topo_sort(topo, new_start, new_end, size, level+1, num_topo_level);
            new_start = i;
            last = topo[i*num_topo_level+level];
        }
    }
    return;
}

bool mca_coll_future_topo_is_mapbycore(int *topo, struct ompi_communicator_t *comm, int num_topo_level){
    int i;
    int size = ompi_comm_size(comm);
    for (i=1; i<size; i++) {
        if (topo[(i-1)*num_topo_level] > topo[i*num_topo_level] || topo[(i-1)*num_topo_level+1] > topo[i*num_topo_level+1]) {
            return false;
            
        }
    }
    return true;
}

int *mca_coll_future_topo_init(struct ompi_communicator_t *comm, mca_coll_future_module_t *future_module, int num_topo_level){
    int size;
    size = ompi_comm_size(comm);
    int *topo;
    if (!((future_module->cached_topo) && (future_module->cached_comm == comm))) {
        if (future_module->cached_topo) {
            free(future_module->cached_topo);
            future_module->cached_topo = NULL;
        }
        topo = (int *)malloc(sizeof(int)*size*num_topo_level);
        //get topo infomation
        mca_coll_future_topo_get(topo, comm, num_topo_level);
        mca_coll_future_topo_print(topo, comm, num_topo_level);
        
        //check if the processes are mapped by core
        future_module->is_mapbycore = mca_coll_future_topo_is_mapbycore(topo, comm, num_topo_level);
        //sort the topo such that each group is contiguous
        if (!future_module->is_mapbycore) {
            mca_coll_future_topo_sort(topo, 0, size-1, size, 0, num_topo_level);
        }
        future_module->cached_topo = topo;
        future_module->cached_comm = comm;
    }
    else {
        topo = future_module->cached_topo;
    }
    
    mca_coll_future_topo_print(topo, comm, num_topo_level);
    return topo;
}

void mca_coll_future_topo_print(int *topo, struct ompi_communicator_t *comm, int num_topo_level){
    int rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    
    if (rank == 0) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: Future Scatter topo: ", rank));
        int i;
        for (i=0; i<size*num_topo_level; i++) {
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "%d ", topo[i]));
        }
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "\n"));

    }
}

