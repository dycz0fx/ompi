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
#include "avx_op_reduce.h"
#include "mpi.h"
#include "ompi/mca/mca.h"
//#include "opal/datatype/opal_convertor.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/communicator/communicator.h"
#include "ompi/win/win.h"
#include "ompi/include/mpi.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/util/info.h"
#include "ompi/op/op.h"
#include "opal/runtime/opal_progress.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_tags.h"

#define MAX_SEG_SIZE 6000000 
//#define MAX_SM_SIZE 70
BEGIN_C_DECLS

    /**
     * Structure to hold the shared coll component.  First it holds the
     * base coll component, and then holds a bunch of
     * shared-coll-component-specific stuff (e.g., current MCA param
     * values).
     */
    typedef struct mca_coll_shared_component_t {
        /** Base coll component */
        mca_coll_base_component_2_0_0_t super;

        /** MCA parameter: Priority of this component */
        int shared_priority;
    } mca_coll_shared_component_t;

    /** Coll shared module */
    typedef struct mca_coll_shared_module_t {
        /** Base module */
	    mca_coll_base_module_t super;

        /* Whether this module has been lazily initialized or not yet */
        bool enabled;

        /* Bcast and Reduce */
        char *sm_data_ptr;   //local shared memory data buf
        MPI_Win sm_data_win;
        char **data_buf;     //address array of global shared memory
        int *sm_ctrl_ptr;
        MPI_Win sm_ctrl_win;
        int **ctrl_buf;

    } mca_coll_shared_module_t;
    OBJ_CLASS_DECLARATION(mca_coll_shared_module_t);

    /**
     * Global component instance
     */
    OMPI_MODULE_DECLSPEC extern mca_coll_shared_component_t mca_coll_shared_component;

    /*
     * coll module functions
     */
    int mca_coll_shared_init_query(bool enable_progress_threads,
			       bool enable_mpi_threads);

    mca_coll_base_module_t *
    mca_coll_shared_comm_query(struct ompi_communicator_t *comm, int *priority);

    /* Lazily enable a module (since it involves expensive/slow mmap
       allocation, etc.) */
    int ompi_coll_shared_lazy_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm);

    int mca_coll_shared_barrier_intra(struct ompi_communicator_t *comm,
				  mca_coll_base_module_t *module);
    int mca_coll_shared_reduce_intra(const void *sbuf, void* rbuf, int count,
				 struct ompi_datatype_t *dtype,
				 struct ompi_op_t *op,
				 int root,
				 struct ompi_communicator_t *comm,
				 mca_coll_base_module_t *module);
    int mca_coll_shared_reduce_shared_ring(const void *sbuf, void* rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

    int mca_coll_shared_reduce_linear(const void *sbuf, void* rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_op_t *op,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);
    int mca_coll_shared_reduce_binomial(const void *sbuf, void* rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);
    int mca_coll_shared_reduce_pipeline(const void *sbuf, void* rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);
    int mca_coll_shared_reduce_generic( const void* sendbuf, void* recvbuf, int original_count,
                                  ompi_datatype_t* datatype, ompi_op_t* op,
                                  int root, ompi_communicator_t* comm,
                                  mca_coll_base_module_t *module,
                                  ompi_coll_tree_t* tree, int count_by_segment,
                                  int max_outstanding_reqs );
    int mca_coll_shared_allreduce_intra(const void *sbuf, void *rbuf,
                                    int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);
    int mca_coll_shared_allreduce_shared_ring(const void *sbuf, void *rbuf,
                                    int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_ring_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_linear_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_linear_nofence_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_binary(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_binomial(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int mca_coll_shared_bcast_pipeline(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
    int ompi_coll_shared_bcast_intra_generic( void* buffer,
                                       int original_count,
                                       struct ompi_datatype_t* datatype,
                                       int root,
                                       struct ompi_communicator_t* comm,
                                       mca_coll_base_module_t *module,
                                       uint32_t count_by_segment,
                                       ompi_coll_tree_t* tree );



END_C_DECLS

#endif /* MCA_COLL_SM_EXPORT_H */
