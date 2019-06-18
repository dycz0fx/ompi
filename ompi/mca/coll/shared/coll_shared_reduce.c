#include "coll_shared.h"

int mca_coll_shared_reduce_intra(const void *sbuf, void* rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module){
    ptrdiff_t extent, lower_bound;
    
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    int size = ompi_comm_size(comm);
    int seg_size = count / size;
    if (count*extent <= 32*1024) {
        mca_coll_shared_reduce_binomial(sbuf, rbuf, count, dtype, op, root, comm, module);
    }
    else if (seg_size*extent > MAX_SEG_SIZE) {
        mca_coll_shared_reduce_pipeline(sbuf, rbuf, count, dtype, op, root, comm, module);
    }
    else{
        mca_coll_shared_reduce_shared_ring(sbuf, rbuf, count, dtype, op, root, comm, module);
    }

    return OMPI_SUCCESS;

}

int mca_coll_shared_reduce_shared_ring(const void *sbuf, void* rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }

    //printf("In shared reduce\n");
    int size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    int i;
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    int seg_size, l_seg_size;
    seg_size = count / size;
    l_seg_size = seg_size;
    if (rank == size - 1) {
        seg_size = count - rank*l_seg_size;
    }
    shared_module->ctrl_buf[rank][0] = rank;
    shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    int cur = rank;
    for (i=0; i<size; i++) {
        if (cur != size-1) {
            seg_size = l_seg_size;
        }
        else {
            seg_size = count - cur*l_seg_size;
        }
        while (rank != shared_module->ctrl_buf[cur][0]) {opal_progress();}
        if (cur == rank) {
            //memcpy(sbuf+cur*l_seg_size, data_buf[cur], seg_size*sizeof(int));
            //for (j=0; j<seg_size; j++) {
            //    shared_module->data_buf[cur][j] = ((char *)sbuf+cur*l_seg_size*extent)[j];
            //}
            memcpy(shared_module->data_buf[cur], (char *)sbuf+cur*l_seg_size*extent, seg_size*extent);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d rank %d]: After First Copy (%d %d)\n", i, cur, rank, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
        }
        else{
            //printf("[%d cur %d rank %d]: Before Op (%d %d)\n", i, cur, rank, data_buf[cur][0], data_buf[cur][1]);
            //for (j=0; j<seg_size; j++) {
            //    shared_module->data_buf[cur][j] = shared_module->data_buf[cur][j] + ((char *)sbuf+cur*l_seg_size*extent)[j];
            //}
            //ompi_op_reduce(op, (char *)sbuf+cur*l_seg_size*extent, shared_module->data_buf[cur], seg_size, dtype);
            avx_op_reduce((char *)sbuf+cur*l_seg_size*extent, shared_module->data_buf[cur], seg_size);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d rank %d]: Op (%d %d)\n", i, cur, rank, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
        }
        cur = (cur-1+size)%size;
        shared_module->ctrl_buf[cur][0] = (shared_module->ctrl_buf[cur][0]+1)%size;
        shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    }
    if (rank == root) {
        char *c;
        c = rbuf;
        for (i=0; i<size; i++) {
            if (i != size-1) {
                seg_size = l_seg_size;
            }
            else {
                seg_size = count - i*l_seg_size;
            }
            memcpy((char*)c, shared_module->data_buf[i], seg_size*extent);
            c = c+seg_size*extent;
        }
    }
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);

    return OMPI_SUCCESS;
}

int mca_coll_shared_reduce_linear(const void *sbuf, void* rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module){
    size_t seg_count = 2048;
    int max_outstanding_reqs = 0;
    int fanout = ompi_comm_size(comm) - 1;
    ompi_coll_tree_t* tree;
    //fanout need to less than 32
    if (fanout > 1) {
        tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    }
    else{
        tree = ompi_coll_base_topo_build_chain(1, comm, root);
    }
    mca_coll_shared_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree, seg_count, max_outstanding_reqs);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int mca_coll_shared_reduce_binomial(const void *sbuf, void* rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_op_t *op,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module){
    size_t seg_count = 2048;
    int max_outstanding_reqs = 0;
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    mca_coll_shared_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree, seg_count, max_outstanding_reqs);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int mca_coll_shared_reduce_pipeline(const void *sbuf, void* rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module){
    size_t seg_count = 2048*4;
    int max_outstanding_reqs = 0;
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_chain(1, comm, root);
    mca_coll_shared_reduce_generic(sbuf, rbuf, count, dtype, op, root, comm, module, tree, seg_count, max_outstanding_reqs);
    ompi_coll_base_topo_destroy_tree(&tree);
    return OMPI_SUCCESS;
}

int mca_coll_shared_reduce_generic( const void* sendbuf, void* recvbuf, int original_count,
                                  ompi_datatype_t* datatype, ompi_op_t* op,
                                  int root, ompi_communicator_t* comm,
                                  mca_coll_base_module_t *module,
                                  ompi_coll_tree_t* tree, int count_by_segment,
                                  int max_outstanding_reqs )
{
    char *inbuf[2] = {NULL, NULL}, *inbuf_free[2] = {NULL, NULL};
    char *accumbuf = NULL, *accumbuf_free = NULL;
    char *local_op_buffer = NULL, *sendtmpbuf = NULL;
    ptrdiff_t extent, size, gap = 0, segment_increment;
    ompi_request_t **sreq = NULL, *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int num_segments, line, ret, segindex, i, rank;
    int recvcount, prevcount, inbi;
    
    /**
     * Determine number of segments and number of elements
     * sent per operation
     */
    ompi_datatype_type_extent( datatype, &extent );
    num_segments = (int)(((size_t)original_count + (size_t)count_by_segment - (size_t)1) / (size_t)count_by_segment);
    segment_increment = (ptrdiff_t)count_by_segment * extent;
    
    sendtmpbuf = (char*) sendbuf;
    if( sendbuf == MPI_IN_PLACE ) {
        sendtmpbuf = (char *)recvbuf;
    }
    
    rank = ompi_comm_rank(comm);
    
    /* non-leaf nodes - wait for children to send me data & forward up
     (if needed) */
    if( tree->tree_nextsize > 0 ) {
        ptrdiff_t real_segment_size;
        
        /* handle non existant recv buffer (i.e. its NULL) and
         protect the recv buffer on non-root nodes */
        accumbuf = (char*)recvbuf;
        if( (NULL == accumbuf) || (root != rank) ) {
            /* Allocate temporary accumulator buffer. */
            size = opal_datatype_span(&datatype->super, original_count, &gap);
            accumbuf_free = (char*)malloc(size);
            if (accumbuf_free == NULL) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            accumbuf = accumbuf_free - gap;
        }
        
        /* If this is a non-commutative operation we must copy
         sendbuf to the accumbuf, in order to simplfy the loops */
        
        if (!ompi_op_is_commute(op) && MPI_IN_PLACE != sendbuf) {
            ompi_datatype_copy_content_same_ddt(datatype, original_count,
                                                (char*)accumbuf,
                                                (char*)sendtmpbuf);
        }
        /* Allocate two buffers for incoming segments */
        real_segment_size = opal_datatype_span(&datatype->super, count_by_segment, &gap);
        inbuf_free[0] = (char*) malloc(real_segment_size);
        if( inbuf_free[0] == NULL ) {
            line = __LINE__; ret = -1; goto error_hndl;
        }
        inbuf[0] = inbuf_free[0] - gap;
        /* if there is chance to overlap communication -
         allocate second buffer */
        if( (num_segments > 1) || (tree->tree_nextsize > 1) ) {
            inbuf_free[1] = (char*) malloc(real_segment_size);
            if( inbuf_free[1] == NULL ) {
                line = __LINE__; ret = -1; goto error_hndl;
            }
            inbuf[1] = inbuf_free[1] - gap;
        }
        
        /* reset input buffer index and receive count */
        inbi = 0;
        recvcount = 0;
        /* for each segment */
        for( segindex = 0; segindex <= num_segments; segindex++ ) {
            prevcount = recvcount;
            /* recvcount - number of elements in current segment */
            recvcount = count_by_segment;
            if( segindex == (num_segments-1) )
                recvcount = original_count - (ptrdiff_t)count_by_segment * (ptrdiff_t)segindex;
            
            /* for each child */
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                /**
                 * We try to overlap communication:
                 * either with next segment or with the next child
                 */
                /* post irecv for current segindex on current child */
                if( segindex < num_segments ) {
                    void* local_recvbuf = inbuf[inbi];
                    if( 0 == i ) {
                        /* for the first step (1st child per segment) and
                         * commutative operations we might be able to irecv
                         * directly into the accumulate buffer so that we can
                         * reduce(op) this with our sendbuf in one step as
                         * ompi_op_reduce only has two buffer pointers,
                         * this avoids an extra memory copy.
                         *
                         * BUT if the operation is non-commutative or
                         * we are root and are USING MPI_IN_PLACE this is wrong!
                         */
                        if( (ompi_op_is_commute(op)) &&
                           !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_recvbuf = accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }
                    
                    ret = MCA_PML_CALL(irecv(local_recvbuf, recvcount, datatype,
                                             tree->tree_next[i],
                                             MCA_COLL_BASE_TAG_REDUCE, comm,
                                             &reqs[inbi]));
                    if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;}
                }
                /* wait for previous req to complete, if any.
                 if there are no requests reqs[inbi ^1] will be
                 MPI_REQUEST_NULL. */
                /* wait on data from last child for previous segment */
                ret = ompi_request_wait(&reqs[inbi ^ 1],
                                        MPI_STATUSES_IGNORE );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                local_op_buffer = inbuf[inbi ^ 1];
                if( i > 0 ) {
                    /* our first operation is to combine our own [sendbuf] data
                     * with the data we recvd from down stream (but only
                     * the operation is commutative and if we are not root and
                     * not using MPI_IN_PLACE)
                     */
                    if( 1 == i ) {
                        if( (ompi_op_is_commute(op)) &&
                           !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment;
                        }
                    }
                    /* apply operation */
                    //ompi_op_reduce(op, local_op_buffer,
                                   //accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                   //recvcount, datatype );
                    avx_op_reduce(local_op_buffer,
                                   accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                   recvcount);
                } else if ( segindex > 0 ) {
                    void* accumulator = accumbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                    if( tree->tree_nextsize <= 1 ) {
                        if( (ompi_op_is_commute(op)) &&
                           !((MPI_IN_PLACE == sendbuf) && (rank == tree->tree_root)) ) {
                            local_op_buffer = sendtmpbuf + (ptrdiff_t)(segindex-1) * (ptrdiff_t)segment_increment;
                        }
                    }
                    //ompi_op_reduce(op, local_op_buffer, accumulator, prevcount,
                    //               datatype );
                    avx_op_reduce(local_op_buffer, accumulator, prevcount); 
                    /* all reduced on available data this step (i) complete,
                     * pass to the next process unless you are the root.
                     */
                    if (rank != tree->tree_root) {
                        /* send combined/accumulated data to parent */
                        ret = MCA_PML_CALL( send( accumulator, prevcount,
                                                 datatype, tree->tree_prev,
                                                 MCA_COLL_BASE_TAG_REDUCE,
                                                 MCA_PML_BASE_SEND_STANDARD,
                                                 comm) );
                        if (ret != MPI_SUCCESS) {
                            line = __LINE__; goto error_hndl;
                        }
                    }
                    
                    /* we stop when segindex = number of segments
                     (i.e. we do num_segment+1 steps for pipelining */
                    if (segindex == num_segments) break;
                }
                
                /* update input buffer index */
                inbi = inbi ^ 1;
            } /* end of for each child */
        } /* end of for each segment */
        
        /* clean up */
        if( inbuf_free[0] != NULL) free(inbuf_free[0]);
        if( inbuf_free[1] != NULL) free(inbuf_free[1]);
        if( accumbuf_free != NULL ) free(accumbuf_free);
    }
    
    /* leaf nodes
     Depending on the value of max_outstanding_reqs and
     the number of segments we have two options:
     - send all segments using blocking send to the parent, or
     - avoid overflooding the parent nodes by limiting the number of
     outstanding requests to max_oustanding_reqs.
     TODO/POSSIBLE IMPROVEMENT: If there is a way to determine the eager size
     for the current communication, synchronization should be used only
     when the message/segment size is smaller than the eager size.
     */
    else {
        
        /* If the number of segments is less than a maximum number of oustanding
         requests or there is no limit on the maximum number of outstanding
         requests, we send data to the parent using blocking send */
        if ((0 == max_outstanding_reqs) ||
            (num_segments <= max_outstanding_reqs)) {
            
            segindex = 0;
            while ( original_count > 0) {
                if (original_count < count_by_segment) {
                    count_by_segment = original_count;
                }
                ret = MCA_PML_CALL( send((char*)sendbuf +
                                         (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                         count_by_segment, datatype,
                                         tree->tree_prev,
                                         MCA_COLL_BASE_TAG_REDUCE,
                                         MCA_PML_BASE_SEND_STANDARD,
                                         comm) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl; }
                segindex++;
                original_count -= count_by_segment;
            }
        }
        
        /* Otherwise, introduce flow control:
         - post max_outstanding_reqs non-blocking synchronous send,
         - for remaining segments
         - wait for a ssend to complete, and post the next one.
         - wait for all outstanding sends to complete.
         */
        else {
            
            int creq = 0;
            
            //sreq = ompi_coll_base_comm_get_reqs(module->base_data, max_outstanding_reqs);
            sreq = (ompi_request_t**)malloc(sizeof(ompi_request_t*) * max_outstanding_reqs);
            //TODO:free
            if (NULL == sreq) { line = __LINE__; ret = -1; goto error_hndl; }
            
            /* post first group of requests */
            for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[segindex]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                original_count -= count_by_segment;
            }
            
            creq = 0;
            while ( original_count > 0 ) {
                /* wait on a posted request to complete */
                ret = ompi_request_wait(&sreq[creq], MPI_STATUS_IGNORE);
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                
                if( original_count < count_by_segment ) {
                    count_by_segment = original_count;
                }
                ret = MCA_PML_CALL( isend((char*)sendbuf +
                                          (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype,
                                          tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          MCA_PML_BASE_SEND_SYNCHRONOUS, comm,
                                          &sreq[creq]) );
                if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
                creq = (creq + 1) % max_outstanding_reqs;
                segindex++;
                original_count -= count_by_segment;
            }
            
            /* Wait on the remaining request to complete */
            ret = ompi_request_wait_all( max_outstanding_reqs, sreq,
                                        MPI_STATUSES_IGNORE );
            if (ret != MPI_SUCCESS) { line = __LINE__; goto error_hndl;  }
        }
    }
    return OMPI_SUCCESS;
    
error_hndl:  /* error handler */
    OPAL_OUTPUT (( ompi_coll_base_framework.framework_output,
                  "ERROR_HNDL: node %d file %s line %d error %d\n",
                  rank, __FILE__, line, ret ));
    (void)line;  // silence compiler warning
    if( inbuf_free[0] != NULL ) free(inbuf_free[0]);
    if( inbuf_free[1] != NULL ) free(inbuf_free[1]);
    if( accumbuf_free != NULL ) free(accumbuf);
    if( NULL != sreq ) {
        ompi_coll_base_free_reqs(sreq, max_outstanding_reqs);
    }
    return ret;
}
