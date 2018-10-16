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

#ifndef MCA_COLL_FUTURE_EXPORT_H
#define MCA_COLL_FUTURE_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/communicator/communicator.h"
#include "ompi/include/mpi.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/info.h"
#include "ompi/op/op.h"
#include "opal/runtime/opal_progress.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_future_trigger.h"

BEGIN_C_DECLS

struct mca_bcast_argu_s {
    void *buff;
    int count;
    struct ompi_datatype_t *dtype;
    int root;
    struct ompi_communicator_t *comm;
    bool noop;
};
typedef struct mca_bcast_argu_s mca_bcast_argu_t;

struct mca_bcast_next_argu_s {
    void *buff;
    int up_seg_count;
    int low_seg_count;
    struct ompi_datatype_t *dtype;
    int root_sm_rank;
    int root_leader_rank;
    struct ompi_communicator_t *up_comm;
    struct ompi_communicator_t *low_comm;
    int num_segments;
    int sm_rank;
    int cur_seg;
    int w_rank;
    int last_seg_count;
};
typedef struct mca_bcast_next_argu_s mca_bcast_next_argu_t;

struct mca_bcast_first_argu_s {
    void *buff;
    int count;
    struct ompi_datatype_t *dtype;
    int root;
    struct ompi_communicator_t *comm;
    int num;
    bool noop;
};
typedef struct mca_bcast_first_argu_s mca_bcast_first_argu_t;

struct mca_bcast_mid_argu_s {
    void *buff;
    int up_seg_count;
    int low_seg_count;
    struct ompi_datatype_t *dtype;
    int root_sm_rank;
    int root_leader_rank;
    struct ompi_communicator_t *up_comm;
    struct ompi_communicator_t *low_comm;
    int up_num;
    int low_num;
    int num_segments;
    int cur_seg;
    int w_rank;
    int last_seg_count;
    bool noop;
};
typedef struct mca_bcast_mid_argu_s mca_bcast_mid_argu_t;

struct mca_allreduce_argu_s {
    mca_coll_task_t *cur_task;
    void *sbuf;
    void *rbuf;
    int up_seg_count;
    int low_seg_count;
    struct ompi_datatype_t *dtype;
    struct ompi_op_t *op;
    int root_up_rank;
    int root_low_rank;
    struct ompi_communicator_t *up_comm;
    struct ompi_communicator_t *low_comm;
    int up_num;
    int low_num;
    int num_segments;
    int cur_seg;
    int w_rank;     //for testing
    int last_seg_count;
    bool noop;
    ompi_request_t *req;
    int *completed;
    int *ongoing;   //ongoing ireduce or ibcast
};
typedef struct mca_allreduce_argu_s mca_allreduce_argu_t;

/**
 * Structure to hold the future coll component.  First it holds the
 * base coll component, and then holds a bunch of
 * future-coll-component-specific stuff (e.g., current MCA param
 * values).
 */
typedef struct mca_coll_future_component_t {
    /** Base coll component */
    mca_coll_base_component_2_0_0_t super;
    
    /** MCA parameter: Priority of this component */
    int future_priority;
    /* whether output the log message */
    int future_output;
    /* up level segment count for bcast */
    int future_bcast_up_count;
    /* low level segment count for bcast */
    int future_bcast_low_count;
    /* up level segment count for allreduce */
    int future_allreduce_up_count;
    /* low level segment count for allreduce */
    int future_allreduce_low_count;
    
} mca_coll_future_component_t;

/** Coll future module */
typedef struct mca_coll_future_module_t {
    /** Base module */
    mca_coll_base_module_t super;
    
    /* Whether this module has been lazily initialized or not yet */
    bool enabled;
    
    struct ompi_communicator_t *cached_comm;
    struct ompi_communicator_t *cached_sm_comm;
    struct ompi_communicator_t *cached_leader_comm;
    int *cached_vranks;
} mca_coll_future_module_t;
OBJ_CLASS_DECLARATION(mca_coll_future_module_t);

/**
 * Global component instance
 */
OMPI_MODULE_DECLSPEC extern mca_coll_future_component_t mca_coll_future_component;

/*
 * coll module functions
 */
int mca_coll_future_init_query(bool enable_progress_threads,
                               bool enable_mpi_threads);

mca_coll_base_module_t *
mca_coll_future_comm_query(struct ompi_communicator_t *comm, int *priority);

/* Lazily enable a module (since it involves expensive/slow mmap
 allocation, etc.) */
int ompi_coll_future_lazy_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm);
int future_request_free(ompi_request_t** request);

/* Bcast */
int mca_coll_future_bcast_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_future_bcast(void *bcast_argu);
int mca_coll_future_nextbcast(void *bcast_next_argu);
void mac_coll_future_set_bcast_argu(mca_bcast_argu_t *argu, void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, bool noop);
void mac_coll_future_set_nextbcast_argu(mca_bcast_next_argu_t *argu, void *buff, int up_seg_count, int low_seg_count, struct ompi_datatype_t *dtype, int root_sm_rank, int root_leader_rank, struct ompi_communicator_t *up_comm, struct ompi_communicator_t *low_comm, int num_segments, int sm_rank, int cur_seg, int w_rank, int last_seg_count);
void mca_coll_future_reset_seg_count(int *up_seg_count, int *low_seg_count, int *count);
void mca_coll_future_get_ranks(int *vranks, int root, int sm_size, int *root_sm_rank, int *root_leader_rank);
int
mca_coll_future_bcast_intra_adapt(void *buff,
                                  int count,
                                  struct ompi_datatype_t *dtype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);
void mac_coll_future_set_first_argu(mca_bcast_first_argu_t *argu, void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, int num, bool noop);
void mac_coll_future_set_mid_argu(mca_bcast_mid_argu_t *argu, void *buff, int up_seg_count, int low_seg_count, struct ompi_datatype_t *dtype, int root_sm_rank, int root_leader_rank, struct ompi_communicator_t *up_comm, struct ompi_communicator_t *low_comm, int up_num, int low_num, int num_segments, int cur_seg, int w_rank, int last_seg_count, bool noop);
int mca_coll_future_first_task(void *task_argu);
int mca_coll_future_mid_task(void *task_argu);
/* Allreduce */
int
mca_coll_future_allreduce_intra(const void *sbuf,
                                void *rbuf,
                                int count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);
int mca_coll_future_sr_task(void *task_argu);
int mca_coll_future_ir_task(void *task_argu);
int mca_coll_future_ib_task(void *task_argu);
int mca_coll_future_sb_task(void *task_argu);
void mac_coll_future_set_argu(mca_allreduce_argu_t *argu,
                              mca_coll_task_t *cur_task,
                              void *sbuf,
                              void *rbuf,
                              int up_seg_count,
                              int low_seg_count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              int root_up_rank,
                              int root_low_rank,
                              struct ompi_communicator_t *up_comm,
                              struct ompi_communicator_t *low_comm,
                              int up_num,
                              int low_num,
                              int num_segments,
                              int cur_seg,
                              int w_rank,
                              int last_seg_count,
                              bool noop,
                              ompi_request_t *req,
                              int *completed,
                              int *ongoing);
END_C_DECLS

#endif /* MCA_COLL_FUTURE_EXPORT_H */
