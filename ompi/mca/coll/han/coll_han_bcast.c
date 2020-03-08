/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

void mac_coll_han_set_bcast_argu(mca_bcast_argu_t *argu, mca_coll_task_t *cur_task, void *buff, int up_seg_count, int low_seg_count, struct ompi_datatype_t *dtype, int root_up_rank, int root_low_rank, struct ompi_communicator_t *up_comm, struct ompi_communicator_t *low_comm, int up_num, int low_num, int num_segments, int cur_seg, int w_rank, int last_seg_count, bool noop){
    argu->cur_task = cur_task;
    argu->buff = buff;
    argu->up_seg_count = up_seg_count;
    argu->low_seg_count = low_seg_count;
    argu->dtype = dtype;
    argu->root_low_rank = root_low_rank;
    argu->root_up_rank = root_up_rank;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->up_num = up_num;
    argu->low_num = low_num;
    argu->num_segments = num_segments;
    argu->cur_seg = cur_seg;
    argu->w_rank = w_rank;
    argu->last_seg_count = last_seg_count;
    argu->noop = noop;
}

int
mca_coll_han_bcast_intra(void *buff,
                                int count,
                                struct ompi_datatype_t *dtype,
                                int root,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_rank;
    w_rank = ompi_comm_rank(comm);
    int up_seg_count = count;
    int low_seg_count = count;
    size_t typelng;
    ompi_datatype_type_size(dtype, &typelng);
    
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    mca_coll_han_comm_create(comm, han_module);
    ompi_communicator_t *low_comm;
    ompi_communicator_t *up_comm;
    /* auto tune is enabled */
    if (mca_coll_han_component.han_auto_tune && mca_coll_han_component.han_auto_tuned != NULL) {
        uint32_t n = han_auto_tuned_get_n(ompi_comm_size(han_module->cached_up_comms[0]));
        uint32_t c = han_auto_tuned_get_c(ompi_comm_size(han_module->cached_low_comms[0]));
        uint32_t m = han_auto_tuned_get_m(typelng * count);
        uint32_t id = n*mca_coll_han_component.han_auto_tune_c*mca_coll_han_component.han_auto_tune_m + c*mca_coll_han_component.han_auto_tune_m + m;
        uint32_t umod = mca_coll_han_component.han_auto_tuned[id].umod;
        uint32_t lmod = mca_coll_han_component.han_auto_tuned[id].lmod;
        uint32_t fs = mca_coll_han_component.han_auto_tuned[id].fs;
        uint32_t ualg = mca_coll_han_component.han_auto_tuned[id].ualg;
        uint32_t us = mca_coll_han_component.han_auto_tuned[id].us;
        /* set up umod */
        up_comm = han_module->cached_up_comms[umod];
        /* set up lmod */
        low_comm = han_module->cached_low_comms[lmod];
        /* set up fs */
        COLL_BASE_COMPUTED_SEGCOUNT((size_t)fs, typelng, up_seg_count);
        low_seg_count = up_seg_count;
        if (umod == 1) {
            /* set up ualg */
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_algorithm = ualg;
            /* set up us */
            ((mca_coll_adapt_module_t *)(up_comm->c_coll->coll_ibcast_module))->adapt_component->adapt_ibcast_segment_size = us;
        }
    }
    else {
        low_comm = han_module->cached_low_comms[mca_coll_han_component.han_bcast_low_module];
        up_comm = han_module->cached_up_comms[mca_coll_han_component.han_bcast_up_module];
        COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_han_component.han_bcast_up_segsize, typelng, up_seg_count);
        COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_han_component.han_bcast_low_segsize, typelng, low_seg_count);
        mca_coll_han_reset_seg_count(&up_seg_count, &low_seg_count, &count);
    }
    
    int max_seg_count = (up_seg_count > low_seg_count) ? up_seg_count : low_seg_count;
    int up_num = max_seg_count / up_seg_count;
    int low_num = max_seg_count / low_seg_count;
    int num_segments = (count + max_seg_count - 1) / max_seg_count;
    OPAL_OUTPUT_VERBOSE((20, mca_coll_han_component.han_output, "In HAN up_count %d low_count %d count %d num_seg %d\n", up_seg_count, low_seg_count, count, num_segments));
    
    int *vranks = han_module->cached_vranks;
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    
    int root_low_rank;
    int root_up_rank;
    mca_coll_han_get_ranks(vranks, root, low_size, &root_low_rank, &root_up_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d]: root_low_rank %d root_up_rank %d\n", w_rank, root_low_rank, root_up_rank));
    
    /* create t0 tasks for the first union segment */
    mca_coll_task_t *t0 = OBJ_NEW(mca_coll_task_t);
    /* setup up t0 task arguments */
    mca_bcast_argu_t *t = malloc(sizeof(mca_bcast_argu_t));
    mac_coll_han_set_bcast_argu(t, t0, (char *)buff, up_seg_count, low_seg_count, dtype, root_up_rank, root_low_rank, up_comm, low_comm, up_num, low_num, num_segments, 0, w_rank, count-(num_segments-1)*max_seg_count, low_rank!=root_low_rank);
    /* init the first task */
    init_task(t0, mca_coll_han_bcast_t0_task, (void *)t);
    issue_task(t0);

    /* create t1 task */
    mca_coll_task_t *t1 = OBJ_NEW(mca_coll_task_t);
    /* setup up t1 task arguments */
    t->cur_task = t1;
    /* init the t1 task */
    init_task(t1, mca_coll_han_bcast_t1_task, (void *)t);
    issue_task(t1);

    while (t->cur_seg <= t->num_segments - 2) {
        /* create t1 task */
        mca_coll_task_t *t1 = OBJ_NEW(mca_coll_task_t);
        /* setup up t1 task arguments */
        t->cur_task = t1;
        t->buff = (char *)t->buff + extent * max_seg_count;
        t->cur_seg = t->cur_seg + 1;
        /* init the t1 task */
        init_task(t1, mca_coll_han_bcast_t1_task, (void *)t);
        issue_task(t1);
    }
    
    free(t);
    
    return OMPI_SUCCESS;
}

/* t0 task: issue and wait for the upper level ibcast of first union segment */
int mca_coll_han_bcast_t0_task(void *task_argu){
    mca_bcast_argu_t *t = (mca_bcast_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d]: in t0 %d\n", t->w_rank, t->cur_seg));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    else {
        int i;
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        ompi_request_t **reqs = malloc(sizeof(ompi_request_t *)*t->up_num);
        for (i=0; i<t->up_num; i++) {
            t->up_comm->c_coll->coll_ibcast((char *)t->buff+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->root_up_rank, t->up_comm, &(reqs[i]), t->up_comm->c_coll->coll_ibcast_module);
        }
        ompi_request_wait_all(t->up_num, reqs, MPI_STATUSES_IGNORE);
        free(reqs);
        return OMPI_SUCCESS;
    }
}

/* t1 task */
int mca_coll_han_bcast_t1_task(void *task_argu){
    int i;
    mca_bcast_argu_t *t = (mca_bcast_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d]: in t1 %d\n", t->w_rank, t->cur_seg));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    int max_seg_count = (t->up_seg_count > t->low_seg_count) ? t->up_seg_count : t->low_seg_count;
    ompi_request_t **reqs = malloc(sizeof(ompi_request_t *)*t->up_num);
    int req_count = 0;

    if (!t->noop) {
        if (t->cur_seg <= t->num_segments-2) {
            if (t->cur_seg == t->num_segments - 2  && t->last_seg_count != max_seg_count) {
                t->up_comm->c_coll->coll_ibcast((char *)t->buff+extent*max_seg_count, t->last_seg_count, t->dtype, t->root_up_rank, t->up_comm, &(reqs[0]), t->up_comm->c_coll->coll_ibcast_module);
                req_count++;
            }
            else {
                for (i=0; i<t->up_num; i++) {
                    t->up_comm->c_coll->coll_ibcast((char *)t->buff+extent*max_seg_count+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->root_up_rank, t->up_comm, &(reqs[i]), t->up_comm->c_coll->coll_ibcast_module);
                    req_count++;
                }
            }
        }
    }
    
    for (i=0; i<t->low_num; i++) {
        t->low_comm->c_coll->coll_bcast((char *)t->buff+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_low_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
    }
    
    if (!t->noop) {
        ompi_request_wait_all(req_count, reqs, MPI_STATUSES_IGNORE);
    }
    free(reqs);
    
    return OMPI_SUCCESS;
}

