#include "coll_shared.h"

int tag = 1;
int id = 0;
int mca_coll_shared_barrier_intra(struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    //printf("shared barrier\n");

    int w_rank = ompi_comm_rank(comm);
    int temp[1];
    temp[0] = 1;
    tag = !tag;
    int n = opal_atomic_add_fetch_32(&(shared_module->barrier_buf[2]), 1);
    id++;
    printf("[%d] rank=%d %d enter barrier %d num_node %d\n", id, w_rank, shared_module->sm_rank, n, shared_module->num_node);
    if (n == shared_module->sm_size){
        shared_module->barrier_buf[!tag] = 0;
        shared_module->barrier_buf[2]=0;
        printf("lock [%d] rank=%d %d accumulate remote\n", id, w_rank, shared_module->sm_rank);
        //shared_module->root_win->w_osc_module->osc_fence(0, shared_module->root_win);
        shared_module->root_win->w_osc_module->osc_lock(MPI_LOCK_EXCLUSIVE, 0, 0, shared_module->root_win);
        printf("accm [%d] rank=%d %d accumulate remote\n", id, w_rank, shared_module->sm_rank);
        shared_module->root_win->w_osc_module->osc_accumulate(&temp, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, shared_module->root_win);
        printf("unlock [%d] rank=%d %d accumulate remote\n", id, w_rank, shared_module->sm_rank);
        //shared_module->root_win->w_osc_module->osc_fence(0, shared_module->root_win);
        shared_module->root_win->w_osc_module->osc_unlock(0, shared_module->root_win);
        printf("after [%d] rank=%d %d accumulate remote\n", id, w_rank, shared_module->sm_rank);
        
    }
    if (w_rank == 0) {
        while (shared_module->barrier_buf[3] < shared_module->num_node) {
            opal_progress();
        }
        printf("ready [%d] rank=%d %d root_ptr %d\n", id, w_rank, shared_module->sm_rank, shared_module->barrier_buf[3]);
        shared_module->barrier_buf[3] = 0;
        int i;
        //shared_module->barrier_buf[tag] = 1;
        for (i=0; i<shared_module->num_node; i++) {
            //printf("lock [%d] rank=%d %d put\n", id, w_rank, shared_module->sm_rank);
            shared_module->leader_win->w_osc_module->osc_lock(MPI_LOCK_EXCLUSIVE, i, 0, shared_module->leader_win);
            //printf("put [%d] rank=%d %d put\n", id, w_rank, shared_module->sm_rank);
            shared_module->leader_win->w_osc_module->osc_put(&temp, 1, MPI_INT, i, tag, 1, MPI_INT, shared_module->leader_win);
            //printf("unlock [%d] rank=%d %d put\n", id, w_rank, shared_module->sm_rank);
            shared_module->leader_win->w_osc_module->osc_unlock(i, shared_module->leader_win);
            //printf("after [%d] rank=%d %d put\n", id, w_rank, shared_module->sm_rank);
        }
    }
    else{
        while (shared_module->barrier_buf[tag] == 0) {
            opal_progress();
        }
    }
    
    printf("[%d] rank=%d %d exit barrier %d %d %d\n", id, w_rank, shared_module->sm_rank, shared_module->barrier_buf[0], shared_module->barrier_buf[1], shared_module->barrier_buf[2]);

    return OMPI_SUCCESS;
}
