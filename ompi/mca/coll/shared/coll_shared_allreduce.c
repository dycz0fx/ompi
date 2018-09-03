#include "coll_shared.h"

int mca_coll_shared_allreduce_intra(const void *sbuf, void *rbuf,
                                    int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module){
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    if (count*extent <= 8*1024) {
        ompi_coll_base_allreduce_intra_recursivedoubling(sbuf, rbuf, count, dtype, op, comm, module);
    }
    else if (count*extent <= 128*1024) {
        ompi_coll_base_allreduce_intra_ring(sbuf, rbuf, count, dtype, op, comm, module);
    }
    else{
        mca_coll_shared_allreduce_shared_ring(sbuf, rbuf, count, dtype, op, comm, module);
    }
    return OMPI_SUCCESS;

}

int mca_coll_shared_allreduce_shared_ring(const void *sbuf, void *rbuf,
                                 int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }

    //printf("In shared allreduce\n");
    int i;
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    int seg_size, l_seg_size;
    seg_size = count / shared_module->sm_size;
    l_seg_size = seg_size;
    if (shared_module->sm_rank == shared_module->sm_size - 1) {
        seg_size = count - shared_module->sm_rank*l_seg_size;
    }
    shared_module->ctrl_buf[shared_module->sm_rank][0] = shared_module->sm_rank;
    shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    int cur = shared_module->sm_rank;
    for (i=0; i<shared_module->sm_size; i++) {
        if (cur != shared_module->sm_size-1) {
            seg_size = l_seg_size;
        }
        else {
            seg_size = count - cur*l_seg_size;
        }
        while (shared_module->sm_rank != shared_module->ctrl_buf[cur][0]) {;}
        if (cur == shared_module->sm_rank) {
            //memcpy(sbuf+cur*l_seg_size, data_buf[cur], seg_size*sizeof(int));
            //for (j=0; j<seg_size; j++) {
            //    shared_module->data_buf[cur][j] = ((char *)sbuf+cur*l_seg_size*extent)[j];
            //}
            memcpy(shared_module->data_buf[cur], (char *)sbuf+cur*l_seg_size*extent, seg_size*extent);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d rank %d]: After First Copy (%d %d)\n", i, cur, shared_module->sm_rank, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
        }
        else{
            //printf("[%d cur %d rank %d]: Before Op (%d %d)\n", i, cur, sm_rank, data_buf[cur][0], data_buf[cur][1]);
            //for (j=0; j<seg_size; j++) {
            //    shared_module->data_buf[cur][j] = shared_module->data_buf[cur][j] + ((char *)sbuf+cur*l_seg_size*extent)[j];
            //}
            ompi_op_reduce(op, (char *)sbuf+cur*l_seg_size*extent, shared_module->data_buf[cur], seg_size, dtype);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d rank %d]: Op (%d %d)\n", i, cur, shared_module->sm_rank, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
        }
        cur = (cur-1+shared_module->sm_size)%shared_module->sm_size;
        shared_module->ctrl_buf[cur][0] = (shared_module->ctrl_buf[cur][0]+1)%shared_module->sm_size;
        shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    }
    char *c;
    c = rbuf;
    for (i=0; i<shared_module->sm_size; i++) {
        if (i != shared_module->sm_size-1) {
            seg_size = l_seg_size;
        }
        else {
            seg_size = count - i*l_seg_size;
        }
        memcpy((char*)c, shared_module->data_buf[i], seg_size*extent);
        c = c+seg_size*extent;
    }
    return OMPI_SUCCESS;
}

