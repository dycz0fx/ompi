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

#ifndef MCA_COLL_FUTURE_TRIGGER_EXPORT_H
#define MCA_COLL_FUTURE_TRIGGER_EXPORT_H

#include "ompi_config.h"
#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/communicator/communicator.h"
#include "ompi/win/win.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/info.h"
#include "ompi/op/op.h"
#include "opal/runtime/opal_progress.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_tags.h"

#define MAX_TASK_NUM 16
#define MAX_FUTURE_NUM 16

typedef int (*task_func_ptr) (void *);

struct mca_coll_task_s {
    opal_object_t  super;
    /* the functhion pointer in the task */
    task_func_ptr func_ptr;
    void *func_argu;
};

typedef struct mca_coll_task_s mca_coll_task_t;

OBJ_CLASS_DECLARATION(mca_coll_task_t);

/* init task */
int init_task(mca_coll_task_t *t, task_func_ptr func_ptr, void *func_argu);

/* issue the task */
int issue_task(mca_coll_task_t *t);

#endif /* MCA_COLL_FUTURE_TRIGGER_EXPORT_H */

