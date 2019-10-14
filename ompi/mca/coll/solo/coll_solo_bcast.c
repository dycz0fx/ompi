/**
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_solo.h"

int mca_coll_solo_bcast_intra(void *buff, int count,
                              struct ompi_datatype_t *dtype, 
                              int root,
                              struct ompi_communicator_t *comm, 
                              mca_coll_base_module_t * module)
{
    if (ompi_datatype_is_contiguous_memory_layout(dtype, count)) {
        mca_coll_solo_bcast_linear_intra_memcpy(buff, count, dtype, root, comm, module);
    }
    else {
        mca_coll_solo_bcast_linear_intra_osc(buff, count, dtype, root, comm, module);
    }
    return OMPI_SUCCESS;
}

int mca_coll_solo_bcast_linear_intra_memcpy(void *buff, int count,
                                            struct ompi_datatype_t *dtype, 
                                            int root,
                                            struct ompi_communicator_t *comm, 
                                            mca_coll_base_module_t * module)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    int id;
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    char *data_buf;
    if ((size_t) count * extent <= mca_coll_solo_component.static_block_size) {
        data_buf = solo_module->data_bufs[root];
    } else if ((size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            id = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, count * extent);
        }
        mca_coll_solo_bcast_linear_intra_memcpy(&id, 1, MPI_INT, root, comm, module);
        data_buf = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, id, 
                                                 count * extent);
    } else {
        /* TODO: Add support for very large messages */
        //printf("TOO BIG\n");
    }

    //solo_module->dynamic_win->w_osc_module->osc_fence(0, solo_module->dynamic_win);
    if (rank == root) {
        memcpy(data_buf, (char *) buff, count * extent);
    }
    //solo_module->dynamic_win->w_osc_module->osc_fence(0, solo_module->dynamic_win);
    mac_coll_solo_barrier_intra(comm, module);
    if (rank != root) {
        memcpy((char *) buff, data_buf, count * extent);
    }
    //solo_module->dynamic_win->w_osc_module->osc_fence(0, solo_module->dynamic_win);
    mac_coll_solo_barrier_intra(comm, module);
    if ((size_t) count * extent > mca_coll_solo_component.static_block_size &&
        (size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, id, count * extent);
        }
    } else {
        /* TODO: Add support for very large messages */
        //printf("TOO BIG\n");
    }
    return OMPI_SUCCESS;
}

int mca_coll_solo_bcast_linear_intra_osc(void *buff, int count,
                                         struct ompi_datatype_t *dtype, 
                                         int root,
                                         struct ompi_communicator_t *comm, 
                                         mca_coll_base_module_t * module)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    int id = 0;
    char **attached_bufs = NULL;
    MPI_Win cur_win;
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    char *data_buf;
    if ((size_t) count * extent <= mca_coll_solo_component.static_block_size) {
        data_buf = (char *) 0 + 4 * opal_cache_line_size;
        cur_win = solo_module->static_win;
    } else if ((size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            id = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, count * extent);
            data_buf = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, id,
                                                     count * extent);
            attached_bufs = mca_coll_solo_attach_buf(solo_module, comm, data_buf, count * extent);
        } else {
            attached_bufs = mca_coll_solo_attach_buf(solo_module, comm, NULL, 0);
        }
        data_buf = attached_bufs[root];
        cur_win = solo_module->dynamic_win;
    } else {
        //printf("TOO BIG\n");
    }

    /* Root copy to shared memory */
    cur_win->w_osc_module->osc_fence(0, cur_win);
    if (rank == root) {
        cur_win->w_osc_module->osc_put(buff, count, dtype, root, (ptrdiff_t) data_buf, count, dtype, 
                                       cur_win);
    }
    cur_win->w_osc_module->osc_fence(0, cur_win);
    /* Other processes copy data from shared memory */
    if (rank != root) {
        cur_win->w_osc_module->osc_get(buff, count, dtype, root, (ptrdiff_t) data_buf, count, dtype,
                                       cur_win);
    }
    cur_win->w_osc_module->osc_fence(0, cur_win);

    if ((size_t) count * extent > mca_coll_solo_component.static_block_size &&
        (size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            mca_coll_solo_detach_buf(solo_module, comm, data_buf, &attached_bufs);
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, id, count * extent);
        } else {
            mca_coll_solo_detach_buf(solo_module, comm, NULL, &attached_bufs);
        }
    } else {
        //printf("TOO BIG\n");
    }

    return OMPI_SUCCESS;
}
