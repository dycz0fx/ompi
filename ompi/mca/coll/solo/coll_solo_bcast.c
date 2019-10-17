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

/* linear bcast with memcpy */
int mca_coll_solo_bcast_linear_intra_memcpy(void *buff, int count,
                                            struct ompi_datatype_t *dtype, 
                                            int root,
                                            struct ompi_communicator_t *comm, 
                                            mca_coll_base_module_t * module)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_buf - shared among all the processes */
    int id;
    char *data_buf;
    if ((size_t) count * extent <= mca_coll_solo_component.static_block_size) {
        data_buf = solo_module->data_bufs[root];
    } else if ((size_t) count * extent <= mca_coll_solo_component.mpool_small_block_size) {
        if (rank == root) {
            id = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, count * extent);
        }
        mca_coll_solo_bcast_linear_intra_memcpy(&id, 1, MPI_INT, root, comm, module);
        data_buf = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, id, 
                                                 count * extent);
    } else {
        return mca_coll_solo_bcast_pipeline_intra_memcpy(buff, count, dtype, root, comm, module, 
                                                         mca_coll_solo_component.mpool_small_block_size);
    }

    /* Root copy data to the shared memory block */
    if (rank == root) {
        memcpy(data_buf, (char *) buff, count * extent);
    }
    mac_coll_solo_barrier_intra(comm, module);
    /* Other processes copy data from the shared memory block */
    if (rank != root) {
        memcpy((char *) buff, data_buf, count * extent);
    }
    mac_coll_solo_barrier_intra(comm, module);
    if ((size_t) count * extent > mca_coll_solo_component.static_block_size &&
        (size_t) count * extent <= mca_coll_solo_component.mpool_large_block_size) {
        if (rank == root) {
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, id, count * extent);
        }
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
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_buf - shared among all the processes */
    int id = 0;
    char **attached_bufs = NULL;
    MPI_Win cur_win;
    char *data_buf;
    if ((size_t) count * extent <= mca_coll_solo_component.static_block_size) {
        data_buf = (char *) 0 + 4 * opal_cache_line_size;
        cur_win = solo_module->static_win;
    } else if ((size_t) count * extent <= mca_coll_solo_component.mpool_small_block_size) {
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
        return mca_coll_solo_bcast_pipeline_intra_osc(buff, count, dtype, root, comm, module, 
                                                      mca_coll_solo_component.mpool_small_block_size);
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
    } 

    return OMPI_SUCCESS;
}

int mca_coll_solo_bcast_pipeline_intra_memcpy(void *buff, int count,
                                              struct ompi_datatype_t *dtype, 
                                              int root,
                                              struct ompi_communicator_t *comm, 
                                              mca_coll_base_module_t * module,
                                              size_t seg_size)
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_bufs - shared among all the processes, needs two for the pipelining */
    int ids[2];
    char *data_bufs[2];
    int i;
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            ids[i] = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, seg_size);
        }
    }
    mca_coll_solo_bcast_linear_intra_memcpy(ids, 2, MPI_INT, root, comm, module);
    for (i = 0; i < 2; i++) {
        data_bufs[i] = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, ids[i], 
                                                     seg_size);
    }
    
    int seg_count = count;
    size_t typelng;
    ompi_datatype_type_size(dtype, &typelng);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, typelng, seg_count);
    int num_segments = (count + seg_count - 1) / seg_count;
    int last_count = count - seg_count * (num_segments - 1);

    for (i = 0; i <= num_segments; i++) {
        int cur = i & 1;
        int pre = !cur;
        if (i == 0) {
            /* In the first iteration, root copies data to the current shared memory block */
            if (rank == root) {
                memcpy(data_bufs[cur], (char *) buff, seg_count * extent);
            }
        }
        else if ( i == num_segments) {
            /* In the last iteration, other processes copy data from the previous shared memory block */
            memcpy(((char *) buff) + seg_count * extent * (i - 1), data_bufs[pre], last_count * extent);
        }
        else {
            /** 
             * For other iterations, root copies data to the current shared memory block and 
             * other proceeses copy data from the previous shared memory block.
             */
            if (rank == root) {
                int temp_count = seg_count;
                if ( i == num_segments - 1) {
                    temp_count = last_count;
                }
                memcpy(data_bufs[cur], ((char *) buff) + seg_count * extent * i, temp_count * extent);
            }
            else {
                memcpy(((char *) buff) + seg_count * extent * (i - 1), data_bufs[pre], seg_count * extent);
            }
        }
        mac_coll_solo_barrier_intra(comm, module);
    }

    /* Return the data_bufs */
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, ids[i], seg_size);
        }
    }
    
    return OMPI_SUCCESS;
}

int mca_coll_solo_bcast_pipeline_intra_osc(void *buff, int count,
                                           struct ompi_datatype_t *dtype, 
                                           int root,
                                           struct ompi_communicator_t *comm, 
                                           mca_coll_base_module_t * module,
                                           size_t seg_size) 
{
    mca_coll_solo_module_t *solo_module = (mca_coll_solo_module_t *) module;

    int rank = ompi_comm_rank(comm);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    /* Enable solo module if necessary */
    if (!solo_module->enabled) {
        mca_coll_solo_lazy_enable(module, comm);
    }
    /* Init the data_bufs - shared among all the processes, needs two for the pipelining */
    int ids[2];
    char **attached_bufs[2];
    MPI_Win cur_win = solo_module->dynamic_win;
    char *data_bufs[2];
    int i;
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            ids[i] = mca_coll_solo_mpool_request(mca_coll_solo_component.solo_mpool, seg_size);
            data_bufs[i] = mca_coll_solo_mpool_calculate(mca_coll_solo_component.solo_mpool, ids[i],
                                                         seg_size);
            attached_bufs[i] = mca_coll_solo_attach_buf(solo_module, comm, data_bufs[i], seg_size);
        }
        else {
            attached_bufs[i] = mca_coll_solo_attach_buf(solo_module, comm, NULL, 0);
        }
        data_bufs[i] = attached_bufs[i][root];
    }
    
    int seg_count = count;
    size_t typelng;
    ompi_datatype_type_size(dtype, &typelng);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, typelng, seg_count);
    int num_segments = (count + seg_count - 1) / seg_count;
    int last_count = count - seg_count * (num_segments - 1);

    cur_win->w_osc_module->osc_fence(0, cur_win);
    for (i = 0; i <= num_segments; i++) {
        int cur = i & 1;
        int pre = !cur;
        if (i == 0) {
            /* In the first iteration, root copies data to the current shared memory block */
            if (rank == root) {
                cur_win->w_osc_module->osc_put(buff, seg_count, dtype, root, (ptrdiff_t) data_bufs[cur],
                                               seg_count, dtype, cur_win);
            }
        }
        else if ( i == num_segments) {
            /* In the last iteration, other processes copy data from the previous shared memory block */
            cur_win->w_osc_module->osc_get(((char *) buff) + seg_count * extent * (i - 1), 
                                           last_count, dtype, root, (ptrdiff_t) data_bufs[pre], 
                                           last_count, dtype, cur_win);
        }
        else {
            /** 
             * For other iterations, root copies data to the current shared memory block and 
             * other proceeses copy data from the previous shared memory block.
             */
            if (rank == root) {
                int temp_count = seg_count;
                if ( i == num_segments - 1) {
                    temp_count = last_count;
                }
                cur_win->w_osc_module->osc_put(((char *) buff) + seg_count * extent * i, 
                                               temp_count, dtype, root, (ptrdiff_t) data_bufs[cur],
                                               temp_count, dtype, cur_win);
            }
            else {
                cur_win->w_osc_module->osc_get(((char *) buff) + seg_count * extent * (i - 1), 
                                               seg_count, dtype, root, (ptrdiff_t) data_bufs[pre], 
                                               seg_count, dtype, cur_win);
            }
        }
        cur_win->w_osc_module->osc_fence(0, cur_win);
    }

    /* Return the data_bufs */
    for (i = 0; i < 2; i++) {
        if (rank == root) {
            mca_coll_solo_detach_buf(solo_module, comm, data_bufs[i], &attached_bufs[i]);
            mca_coll_solo_mpool_return(mca_coll_solo_component.solo_mpool, ids[i], seg_size);
        }
        else {
            mca_coll_solo_detach_buf(solo_module, comm, NULL, &attached_bufs[i]);
        }
    }
    
    return OMPI_SUCCESS;
}
