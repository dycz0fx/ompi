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
    int var_id;
    ompi_communicator_t *sm_comm;
    int tmp_priority = 60;
    const int *tmp = NULL;
    int error;
    error = mca_base_var_find_by_name("coll_sm_priority", &var_id);
    error = mca_base_var_get_value(var_id, &tmp, NULL, NULL);
    printf("sm_priority %d %d\n", *tmp, error);
    error = mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
    error = mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
    error = mca_base_var_get_value(var_id, &tmp, NULL, NULL);
    printf("sm_priority %d %d\n", *tmp, error);
    ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &sm_comm);
    tmp_priority = 20;
    error = mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
    error = mca_base_var_get_value(var_id, &tmp, NULL, NULL);
    printf("sm_priority %d %d\n", *tmp, error);
    int sm_size = ompi_comm_size(sm_comm);
    int sm_rank = ompi_comm_rank(sm_comm);
    
    /* create leader_comm which contain one process per node (across nodes) */
    ompi_communicator_t *leader_comm;
    tmp_priority = 60;
    error = mca_base_var_find_by_name("coll_tuned_priority", &var_id);
    error = mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
    error = mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
    error = mca_base_var_get_value(var_id, &tmp, NULL, NULL);
    printf("tuned_priority %d %d\n", *tmp, error);
    ompi_comm_split(comm, sm_rank, w_rank, &leader_comm, false);
    tmp_priority = 30;
    error = mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
    error = mca_base_var_get_value(var_id, &tmp, NULL, NULL);
    printf("tuned_priority %d %d\n", *tmp, error);

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
        init_task(up_b, mca_coll_future_bcast_wrapper, (void *)(&up_task_argu));
        
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
        init_task(low_b0, mca_coll_future_bcast_wrapper, (void *)(&low_task_argu0));
        
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
        init_task(low_b1, mca_coll_future_bcast_wrapper, (void *)(&low_task_argu1));
        
        add_butterfly(up_b, f);
        printf("[%d]: add butterfly task %p future %p\n", w_rank, (void *)up_b, (void *)f);
        add_tornado(low_b0, f);
        printf("[%d]: add tornado task %p future %p\n", w_rank, (void *)low_b0, (void *)f);
        add_tornado(low_b1, f);
        printf("[%d]: add tornado task %p future %p\n", w_rank, (void *)low_b1, (void *)f);
        execute_task(up_b);
    }
    
    return OMPI_SUCCESS;
}

int mca_coll_future_bcast_wrapper(void *bcast_argu){
    mca_bcast_argu_t *t = (mca_bcast_argu_t *)bcast_argu;
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    else {
        t->comm->c_coll->coll_bcast(t->buff, t->count, t->dtype, t->root, t->comm, t->comm->c_coll->coll_bcast_module);
        /*
        size_t seg_count = 2048;
        ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(t->comm, t->root);
        ompi_coll_base_bcast_intra_generic(t->buff, t->count, t->dtype, t->root, t->comm, t->module, seg_count, tree);
        ompi_coll_base_topo_destroy_tree(&tree);
         */
        return OMPI_SUCCESS;
        
    }
}
