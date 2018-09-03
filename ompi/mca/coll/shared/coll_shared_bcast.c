#include "coll_shared.h"

int mca_coll_shared_bcast_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    if (count*extent <= 2048) {
        mca_coll_shared_bcast_binomial(buff, count, dtype, root, comm, module);
    }
    else{
        mca_coll_shared_bcast_linear_intra(buff, count, dtype, root, comm, module);
    }
    return OMPI_SUCCESS;
}

int mca_coll_shared_bcast_ring_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    
    int i;
    int w_rank = ompi_comm_rank(comm);
    int v_rank = w_rank;
    if (w_rank == root) {
        v_rank = 0;
    }
    else if (w_rank < root) {
        v_rank = w_rank + 1;
    }
    else {
        v_rank = w_rank;
    }
    //printf("In shared bcast w_rank %d v_rank %d\n", w_rank, v_rank);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    int seg_size, l_seg_size;
    seg_size = count / (shared_module->sm_size-1);
    l_seg_size = seg_size;
    if (v_rank == shared_module->sm_size-1) {
        seg_size = count - (shared_module->sm_size-2) * l_seg_size;
    }
    //root copy data to shared memory
    if (v_rank == 0) {
        char *c;
        c = buff;
        for (i=1; i<shared_module->sm_size; i++) {
            if (i != shared_module->sm_size-1) {
                seg_size = l_seg_size;
            }
            else {
                seg_size = count - (shared_module->sm_size-2)*l_seg_size;
            }
            memcpy(shared_module->data_buf[i], c, seg_size*extent);
            c = c+seg_size*extent;
        }
        
    }

    shared_module->ctrl_buf[v_rank][0] = v_rank;
    shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    
    int cur = v_rank;
    if (v_rank > 0) {
        for (i=1; i<shared_module->sm_size; i++) {
            if (cur != shared_module->sm_size-1) {
                seg_size = l_seg_size;
            }
            else {
                seg_size = count - (shared_module->sm_size-2)*l_seg_size;
            }
            while (v_rank != shared_module->ctrl_buf[cur][0]) {;}
            memcpy((char *)buff+(cur-1)*l_seg_size*extent, shared_module->data_buf[cur], seg_size*extent);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d v_rank %d]: Copy %d (%d %d)\n", i, cur, v_rank, seg_size, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
            cur = (cur-2+shared_module->sm_size-1)%(shared_module->sm_size-1)+1;
            shared_module->ctrl_buf[cur][0] = (shared_module->ctrl_buf[cur][0])%(shared_module->sm_size-1)+1;
            shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
        }
    }
    else {
        for (i=1; i<shared_module->sm_size; i++) {
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
        }

    }
    return OMPI_SUCCESS;
}


int mca_coll_shared_bcast_linear_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    
    //printf("In shared linear bcast\n");
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    int w_rank = ompi_comm_rank(comm);
    if (w_rank == root) {
        memcpy(shared_module->data_buf[root], (char*)buff, count*extent);
    }
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);

    if (w_rank != root) {
        memcpy((char*)buff, shared_module->data_buf[root], count*extent);
    }
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    return OMPI_SUCCESS;
}

int mca_coll_shared_bcast_linear_nofence_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    
    //printf("In shared linear bcast\n");
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    int w_rank = ompi_comm_rank(comm);
    if (w_rank == root) {
        memcpy(shared_module->data_buf[root], (char*)buff, count*extent);
    }
    //bcast
    int r[1];
    mca_coll_shared_bcast_binary(r, 1, MPI_INT, root, comm, module);
    if (w_rank != root) {
        memcpy((char*)buff, shared_module->data_buf[root], count*extent);
    }
    //barrier
    ompi_coll_base_barrier_intra_recursivedoubling(comm, module);
    //shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    return OMPI_SUCCESS;
}

int mca_coll_shared_bcast_binary(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    size_t seg_count = 2048;
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_tree(2, comm, root);
    ompi_coll_shared_bcast_intra_generic(buff, count, dtype, root, comm, module, seg_count, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int mca_coll_shared_bcast_binomial(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    size_t seg_count = 2048;
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    ompi_coll_shared_bcast_intra_generic(buff, count, dtype, root, comm, module, seg_count, tree);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int ompi_coll_shared_bcast_intra_generic( void* buffer,
                                   int original_count,
                                   struct ompi_datatype_t* datatype,
                                   int root,
                                   struct ompi_communicator_t* comm,
                                   mca_coll_base_module_t *module,
                                   uint32_t count_by_segment,
                                   ompi_coll_tree_t* tree )
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
