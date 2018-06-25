#include "coll_future.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_future_trigger.h"

int mca_coll_future_bcast_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    printf("In Future\n");
    int i;
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_size, w_rank;
    w_size = ompi_comm_size(comm);
    w_rank = ompi_comm_rank(comm);
    size_t seg_count = 1024;
    int num_segments = (count + seg_count - 1) / seg_count;
    /* create sm_comm which contain all the process on a node */
    ompi_communicator_t *sm_comm;
    ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &sm_comm);
    int sm_size = ompi_comm_size(sm_comm);
    int sm_rank = ompi_comm_rank(sm_comm);
    /* create leader_comm which contain one process per node (across nodes) */
    ompi_communicator_t *leader_comm;
    ompi_comm_split(comm, sm_rank, w_rank, &leader_comm, false);
    for (i=0; i<num_segments; i++) {
        /* create future */
        mca_coll_future_t *f = OBJ_NEW(mca_coll_future_t);
        /* create task */
        mca_coll_task_t *up_b = OBJ_NEW(mca_coll_task_t);
        /* set up up task arguments */ //for now, root has to 0//
        mca_bcast_argu_t up_task_argu;
        up_task_argu.count = count;
        up_task_argu.buff = buff;
        up_task_argu.dtype = dtype;
        up_task_argu.root = 0;
        up_task_argu.comm = leader_comm;
        up_task_argu.module = module;
        if (sm_rank == 0) {
            up_task_argu.noop = false;
        }
        else {
            up_task_argu.noop = true;
        }
        /* init the up task */
        init_task(up_b, wrap_bcast_binomial, (void *)(&up_task_argu));
        
        mca_coll_task_t *low_b0 = OBJ_NEW(mca_coll_task_t);
        /* set up low task arguments */ //for now, root has to 0//
        mca_bcast_argu_t low_task_argu0;
        low_task_argu0.count = count/2;
        low_task_argu0.buff = buff;
        low_task_argu0.dtype = dtype;
        low_task_argu0.root = 0;
        low_task_argu0.comm = sm_comm;
        low_task_argu0.module = module;
        low_task_argu0.noop = false;
        /* init the low task */
        init_task(low_b0, wrap_bcast_binomial, (void *)(&low_task_argu0));
        
        mca_coll_task_t *low_b1 = OBJ_NEW(mca_coll_task_t);
        /* set up low task arguments */ //for now, root has to 0//
        mca_bcast_argu_t low_task_argu1;
        low_task_argu1.count = count/2;
        low_task_argu1.buff = (char *)buff+extent*low_task_argu1.count;
        low_task_argu1.dtype = dtype;
        low_task_argu1.root = 0;
        low_task_argu1.comm = sm_comm;
        low_task_argu1.module = module;
        low_task_argu1.noop = false;
        /* init the low task */
        init_task(low_b1, wrap_bcast_binomial, (void *)(&low_task_argu1));
        
        add_left(up_b, f);
        add_right(low_b0, f);
        add_right(low_b1, f);
        execute_task(up_b);
        
    }
    
    //mca_coll_future_bcast_binomial(buff, count, dtype, root, comm, module);
    return OMPI_SUCCESS;
}

int wrap_bcast_binomial(void *bcast_argu){
    mca_bcast_argu_t *t = (mca_bcast_argu_t *)bcast_argu;
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    else {
        return mca_coll_future_bcast_binomial(t->buff, t->count, t->dtype, t->root, t->comm, t->module);
    }
    
}

int mca_coll_future_bcast_binomial(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    size_t seg_count = 2048;
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    mca_coll_future_bcast_intra_generic(buff, count, dtype, root, comm, module, seg_count, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int mca_coll_future_bcast_intra_generic(void* buffer, int original_count, struct ompi_datatype_t* datatype, int root, struct ompi_communicator_t* comm, mca_coll_base_module_t *module, uint32_t count_by_segment, ompi_coll_tree_t* tree)
{
    int err = 0, line, i, rank, segindex, req_index;
    int num_segments; /* Number of segments */
    int sendcount;    /* number of elements sent in this segment */
    size_t realsegsize, type_size;
    char *tmpbuf;
    ptrdiff_t extent, lb;
    ompi_request_t *recv_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    ompi_request_t **send_reqs = NULL;
    
#if OPAL_ENABLE_DEBUG
    int size;
    size = ompi_comm_size(comm);
    assert( size > 1 );
#endif
    rank = ompi_comm_rank(comm);
    
    ompi_datatype_get_extent (datatype, &lb, &extent);
    ompi_datatype_type_size( datatype, &type_size );
    num_segments = (original_count + count_by_segment - 1) / count_by_segment;
    realsegsize = (ptrdiff_t)count_by_segment * extent;
    
    /* Set the buffer pointers */
    tmpbuf = (char *) buffer;
    
    if( tree->tree_nextsize != 0 ) {
        send_reqs = (ompi_request_t**)malloc(sizeof(ompi_request_t*) * tree->tree_nextsize);
        if( NULL == send_reqs ) { err = OMPI_ERR_OUT_OF_RESOURCE; line = __LINE__; goto error_hndl; }
    }
    
    /* Root code */
    if( rank == root ) {
        /*
         For each segment:
         - send segment to all children.
         The last segment may have less elements than other segments.
         */
        sendcount = count_by_segment;
        for( segindex = 0; segindex < num_segments; segindex++ ) {
            if( segindex == (num_segments - 1) ) {
                sendcount = original_count - segindex * count_by_segment;
            }
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                err = MCA_PML_CALL(isend(tmpbuf, sendcount, datatype,
                                         tree->tree_next[i],
                                         MCA_COLL_BASE_TAG_BCAST,
                                         MCA_PML_BASE_SEND_STANDARD, comm,
                                         &send_reqs[i]));
                if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            }
            
            /* complete the sends before starting the next sends */
            err = ompi_request_wait_all( tree->tree_nextsize, send_reqs,
                                        MPI_STATUSES_IGNORE );
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            
            /* update tmp buffer */
            tmpbuf += realsegsize;
            
        }
    }
    
    /* Intermediate nodes code */
    else if( tree->tree_nextsize > 0 ) {
        /*
         Create the pipeline.
         1) Post the first receive
         2) For segments 1 .. num_segments
         - post new receive
         - wait on the previous receive to complete
         - send this data to children
         3) Wait on the last segment
         4) Compute number of elements in last segment.
         5) Send the last segment to children
         */
        req_index = 0;
        err = MCA_PML_CALL(irecv(tmpbuf, count_by_segment, datatype,
                                 tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
                                 comm, &recv_reqs[req_index]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
        
        for( segindex = 1; segindex < num_segments; segindex++ ) {
            
            req_index = req_index ^ 0x1;
            
            /* post new irecv */
            err = MCA_PML_CALL(irecv( tmpbuf + realsegsize, count_by_segment,
                                     datatype, tree->tree_prev,
                                     MCA_COLL_BASE_TAG_BCAST,
                                     comm, &recv_reqs[req_index]));
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            
            /* wait for and forward the previous segment to children */
            err = ompi_request_wait( &recv_reqs[req_index ^ 0x1],
                                    MPI_STATUS_IGNORE );
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                err = MCA_PML_CALL(isend(tmpbuf, count_by_segment, datatype,
                                         tree->tree_next[i],
                                         MCA_COLL_BASE_TAG_BCAST,
                                         MCA_PML_BASE_SEND_STANDARD, comm,
                                         &send_reqs[i]));
                if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            }
            
            /* complete the sends before starting the next iteration */
            err = ompi_request_wait_all( tree->tree_nextsize, send_reqs,
                                        MPI_STATUSES_IGNORE );
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            
            /* Update the receive buffer */
            tmpbuf += realsegsize;
            
        }
        
        /* Process the last segment */
        err = ompi_request_wait( &recv_reqs[req_index], MPI_STATUS_IGNORE );
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
        sendcount = original_count - (ptrdiff_t)(num_segments - 1) * count_by_segment;
        for( i = 0; i < tree->tree_nextsize; i++ ) {
            err = MCA_PML_CALL(isend(tmpbuf, sendcount, datatype,
                                     tree->tree_next[i],
                                     MCA_COLL_BASE_TAG_BCAST,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     &send_reqs[i]));
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
        }
        
        err = ompi_request_wait_all( tree->tree_nextsize, send_reqs,
                                    MPI_STATUSES_IGNORE );
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
    }
    
    /* Leaf nodes */
    else {
        /*
         Receive all segments from parent in a loop:
         1) post irecv for the first segment
         2) for segments 1 .. num_segments
         - post irecv for the next segment
         - wait on the previous segment to arrive
         3) wait for the last segment
         */
        req_index = 0;
        err = MCA_PML_CALL(irecv(tmpbuf, count_by_segment, datatype,
                                 tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
                                 comm, &recv_reqs[req_index]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
        
        for( segindex = 1; segindex < num_segments; segindex++ ) {
            req_index = req_index ^ 0x1;
            tmpbuf += realsegsize;
            /* post receive for the next segment */
            err = MCA_PML_CALL(irecv(tmpbuf, count_by_segment, datatype,
                                     tree->tree_prev, MCA_COLL_BASE_TAG_BCAST,
                                     comm, &recv_reqs[req_index]));
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
            /* wait on the previous segment */
            err = ompi_request_wait( &recv_reqs[req_index ^ 0x1],
                                    MPI_STATUS_IGNORE );
            if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
        }
        
        err = ompi_request_wait( &recv_reqs[req_index], MPI_STATUS_IGNORE );
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
    }
    
    return (MPI_SUCCESS);
    
error_hndl:
    OPAL_OUTPUT( (ompi_coll_base_framework.framework_output,"%s:%4d\tError occurred %d, rank %2d",
                  __FILE__, line, err, rank) );
    (void)line;  // silence compiler warnings
    ompi_coll_base_free_reqs( recv_reqs, 2);
    if( NULL != send_reqs ) {
        ompi_coll_base_free_reqs(send_reqs, tree->tree_nextsize);
    }
    
    return err;
}
