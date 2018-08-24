#include "coll_future.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_future_trigger.h"

void mac_coll_future_set_bcast_argu(mca_bcast_argu_t *argu, void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, bool noop){
    argu->buff = buff;
    argu->count = count;
    argu->dtype = dtype;
    argu->root = root;
    argu->comm = comm;
    argu->noop = noop;
}

void mac_coll_future_set_nextbcast_argu(mca_bcast_next_argu_t *argu, void *buff, int up_seg_count, int low_seg_count, struct ompi_datatype_t *dtype, int root_sm_rank, int root_leader_rank ,struct ompi_communicator_t *up_comm, struct ompi_communicator_t *low_comm, int num_segments, int sm_rank, int cur_seg, int w_rank, int last_seg_count){
    argu->buff = buff;
    argu->up_seg_count = up_seg_count;
    argu->low_seg_count = low_seg_count;
    argu->dtype = dtype;
    argu->root_sm_rank = root_sm_rank;
    argu->root_leader_rank = root_leader_rank;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->num_segments = num_segments;
    argu->sm_rank = sm_rank;
    argu->cur_seg = cur_seg;
    argu->w_rank = w_rank;
    argu->last_seg_count = last_seg_count;
}

void mac_coll_future_set_first_argu(mca_bcast_first_argu_t *argu, void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, int num, bool noop){
    argu->buff = buff;
    argu->count = count;
    argu->dtype = dtype;
    argu->root = root;
    argu->comm = comm;
    argu->num = num;
    argu->noop = noop;
}

void mac_coll_future_set_mid_argu(mca_bcast_mid_argu_t *argu, void *buff, int up_seg_count, int low_seg_count, struct ompi_datatype_t *dtype, int root_sm_rank, int root_leader_rank, struct ompi_communicator_t *up_comm, struct ompi_communicator_t *low_comm, int up_num, int low_num, int num_segments, int cur_seg, int w_rank, int last_seg_count, bool noop){
    argu->buff = buff;
    argu->up_seg_count = up_seg_count;
    argu->low_seg_count = low_seg_count;
    argu->dtype = dtype;
    argu->root_sm_rank = root_sm_rank;
    argu->root_leader_rank = root_leader_rank;
    argu->up_comm = up_comm;
    argu->low_comm = low_comm;
    argu->up_num = up_num;
    argu->low_num = low_num;
    argu->num_segments = num_segments;
    argu->cur_seg = cur_seg;
    argu->w_rank = w_rank;
    argu->last_seg_count = last_seg_count;
    argu->noop = noop;
}

/* Build multilevel bcast based on blocking bcast
 * | B0 B1 | B3 B4 |
 * |   B2  |   B5  |
 * B0 and B1 will trigger B2 and next_bcast task.
 * In the next_bcast task, it will issue B3, B4 and B5.
*/
int
mca_coll_future_bcast_intra(void *buff,
                            int count,
                            struct ompi_datatype_t *dtype,
                            int root,
                            struct ompi_communicator_t *comm,
                            mca_coll_base_module_t *module)
{
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_size, w_rank, i;
    w_size = ompi_comm_size(comm);
    w_rank = ompi_comm_rank(comm);
    int up_seg_count = mca_coll_future_component.future_up_count;
    int low_seg_count = mca_coll_future_component.future_low_count;
    mca_coll_future_reset_seg_count(&up_seg_count, &low_seg_count, &count);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "In Future %d %d %d\n", up_seg_count, low_seg_count, count));
    int max_seg_count = (up_seg_count > low_seg_count) ? up_seg_count : low_seg_count;
    int up_num = max_seg_count / up_seg_count;
    int low_num = max_seg_count / low_seg_count;
    if (up_num > MAX_TASK_NUM || low_num > MAX_TASK_NUM) {
        return OMPI_ERROR;
    }
    int num_segments = (count + max_seg_count - 1) / max_seg_count;

    ompi_communicator_t *sm_comm;
    ompi_communicator_t *leader_comm;
    int *vranks;
    int sm_rank, sm_size;
    int leader_rank, leader_size;
    mca_coll_future_module_t *future_module = (mca_coll_future_module_t *)module;
    /* use cached communicators if possible */
    if (future_module->cached_comm == comm && future_module->cached_sm_comm != NULL && future_module->cached_leader_comm != NULL && future_module->cached_vranks != NULL) {
        sm_comm = future_module->cached_sm_comm;
        leader_comm = future_module->cached_leader_comm;
        vranks = future_module->cached_vranks;
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);
        leader_size = ompi_comm_size(leader_comm);
        leader_rank = ompi_comm_rank(leader_comm);
    }
    /* create communicators if there is no cached communicator */
    else {
        /* create sm_comm which contain all the process on a node */
        int var_id;
        int tmp_priority = 60;
        const int *origin_priority = NULL;
        int tmp_origin = 0;
        //const int *tmp = NULL;
        mca_base_var_find_by_name("coll_sm_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] sm_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("sm_priority after set %d %d\n", *tmp);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &sm_comm);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] sm_priority set back %d\n", w_rank, *tmp);
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);

        /* create leader_comm which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_tuned_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] tuned_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("tuned_priority after set %d %d\n", *tmp);
        ompi_comm_split(comm, sm_rank, w_rank, &leader_comm, false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] tuned_priority set back %d\n", w_rank, *tmp);
        leader_size = ompi_comm_size(leader_comm);
        leader_rank = ompi_comm_rank(leader_comm);

        vranks = malloc(sizeof(int) * w_size);
        /* do allgather to gather vrank from each process so every process will know other processes vrank*/
        int vrank = sm_size * leader_rank + sm_rank;
        comm->c_coll->coll_allgather(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm, comm->c_coll->coll_allgather_module);
        future_module->cached_comm = comm;
        future_module->cached_sm_comm = sm_comm;
        future_module->cached_leader_comm = leader_comm;
        future_module->cached_vranks = vranks;
    }
    
    int root_sm_rank;
    int root_leader_rank;
    mca_coll_future_get_ranks(vranks, root, sm_size, &root_sm_rank, &root_leader_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: root_sm_rank %d root_leader_rank %d\n", w_rank, root_sm_rank, root_leader_rank));

    /* create future */
    mca_coll_future_t *f = OBJ_NEW(mca_coll_future_t);
    
    mca_coll_task_t **up_list = malloc(sizeof(mca_coll_task_t *)*up_num);
    /* create upper level bcast tasks for the first union segment */
    for (i=0; i<up_num; i++) {
        up_list[i] = OBJ_NEW(mca_coll_task_t);
        /* setup up task arguments */ //for now, root has to 0//
        mca_bcast_argu_t *up_task_argu = malloc(sizeof(mca_bcast_argu_t));
        mac_coll_future_set_bcast_argu(up_task_argu, (char *)buff+extent*up_seg_count*i, up_seg_count, dtype, root_leader_rank, leader_comm, sm_rank!=root_sm_rank);
        /* init the up task */
        init_task(up_list[i], mca_coll_future_bcast, (void *)(up_task_argu));
        /* add the up task into future, up task will trigger the future */
        add_butterfly(up_list[i], f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add butterfly task %p future %p\n", w_rank, (void *)up_list[i], (void *)f));
    }
    
    /* create lower level bcast tasks for the first union segment */
    for (i=0; i<low_num; i++) {
        mca_coll_task_t *low = OBJ_NEW(mca_coll_task_t);
        /* set up low task arguments */ //for now, root has to 0//
        mca_bcast_argu_t *low_task_argu = malloc(sizeof(mca_bcast_argu_t));
        mac_coll_future_set_bcast_argu(low_task_argu, (char *)buff+extent*low_seg_count*i, low_seg_count, dtype, root_sm_rank, sm_comm, false);
        /* init the low task */
        init_task(low, mca_coll_future_bcast, (void *)(low_task_argu));
        /* add the low task into future, low task will be triggered by the future */
        add_tornado(low, f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task %p future %p\n", w_rank, (void *)low, (void *)f));
    }
    
    /* create bcast for next union segment if needed */
    if (num_segments > 1) {
        mca_coll_task_t *next = OBJ_NEW(mca_coll_task_t);
        /* set up next task arguments */ //for now, root has to 0//
        mca_bcast_next_argu_t *next_task_argu = malloc(sizeof(mca_bcast_next_argu_t));
        mac_coll_future_set_nextbcast_argu(next_task_argu, (char *)buff+extent*max_seg_count, up_seg_count, low_seg_count, dtype, root_sm_rank, root_leader_rank, leader_comm, sm_comm, num_segments, sm_rank, 1, w_rank, count-(num_segments-1)*max_seg_count);
        /* init the next task */
        init_task(next, mca_coll_future_nextbcast, (void *)(next_task_argu));
        /* add the nextbcast task into future, nextbcast task will be triggered by the future */
        add_tornado(next, f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task %p future %p\n", w_rank, (void *)next, (void *)f));
    }

    for (i=0; i<up_num; i++) {
        execute_task(up_list[i]);
    }
    
    free(up_list);
    return OMPI_SUCCESS;
}

int mca_coll_future_bcast(void *bcast_argu){
    mca_bcast_argu_t *t = (mca_bcast_argu_t *)bcast_argu;
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    else {
        t->comm->c_coll->coll_bcast(t->buff, t->count, t->dtype, t->root, t->comm, t->comm->c_coll->coll_bcast_module);
        return OMPI_SUCCESS;
    }
}

int mca_coll_future_nextbcast(void *bcast_next_argu){
    /* create future */
    mca_coll_future_t *f = OBJ_NEW(mca_coll_future_t);

    ptrdiff_t extent, lb;
    mca_bcast_next_argu_t *argu = (mca_bcast_next_argu_t *)bcast_next_argu;
    ompi_datatype_get_extent(argu->dtype, &lb, &extent);
    
    int max_seg_count = (argu->up_seg_count > argu->low_seg_count) ? argu->up_seg_count : argu->low_seg_count;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: %d %d %d %d\n", argu->w_rank, argu->num_segments, argu->cur_seg, argu->last_seg_count, max_seg_count));
    /* handle last segment */
    if (argu->num_segments == argu->cur_seg + 1 && argu->last_seg_count < max_seg_count) {
        mca_coll_task_t *up = OBJ_NEW(mca_coll_task_t);
        /* set up up task arguments */ //for now, root has to 0//
        mca_bcast_argu_t *up_task_argu = malloc(sizeof(mca_bcast_argu_t));
        mac_coll_future_set_bcast_argu(up_task_argu, argu->buff, argu->last_seg_count, argu->dtype, argu->root_leader_rank, argu->up_comm, argu->sm_rank!=argu->root_sm_rank);
        /* init the up task */
        init_task(up, mca_coll_future_bcast, (void *)(up_task_argu));
        add_butterfly(up, f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: at last, add butterfly task %p future %p when seg %d\n", argu->w_rank, (void *)up, (void *)f, argu->cur_seg));

        mca_coll_task_t *low = OBJ_NEW(mca_coll_task_t);
        /* set up low task arguments */ //for now, root has to 0//
        mca_bcast_argu_t *low_task_argu = malloc(sizeof(mca_bcast_argu_t));
        mac_coll_future_set_bcast_argu(low_task_argu, argu->buff, argu->last_seg_count, argu->dtype, argu->root_sm_rank, argu->low_comm, false);
        /* init the low task */
        init_task(low, mca_coll_future_bcast, (void *)(low_task_argu));
        add_tornado(low, f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: at last, add tornado task %p future %p when seg %d\n", argu->w_rank, (void *)low, (void *)f, argu->cur_seg));
        execute_task(up);
    }
    /* other than last segment */
    else {
        int i;
        int up_num = max_seg_count / argu->up_seg_count;
        mca_coll_task_t **up_list = malloc(sizeof(mca_coll_task_t *)*up_num);
        for (i=0; i<up_num; i++) {
            up_list[i] = OBJ_NEW(mca_coll_task_t);
            /* set up up task arguments */ //for now, root has to 0//
            mca_bcast_argu_t *up_task_argu = malloc(sizeof(mca_bcast_argu_t));
            mac_coll_future_set_bcast_argu(up_task_argu, (char *)argu->buff+extent*argu->up_seg_count*i, argu->up_seg_count, argu->dtype, argu->root_leader_rank, argu->up_comm, argu->sm_rank!=argu->root_sm_rank);
            /* init the up task */
            init_task(up_list[i], mca_coll_future_bcast, (void *)(up_task_argu));
            add_butterfly(up_list[i], f);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add butterfly task %p future %p when seg %d\n", argu->w_rank, (void *)up_list[i], (void *)f, argu->cur_seg));
        }
        
        int low_num = max_seg_count / argu->low_seg_count;
        for (i=0; i<low_num; i++) {
            mca_coll_task_t *low = OBJ_NEW(mca_coll_task_t);
            /* set up low task arguments */ //for now, root has to 0//
            mca_bcast_argu_t *low_task_argu = malloc(sizeof(mca_bcast_argu_t));
            mac_coll_future_set_bcast_argu(low_task_argu, (char *)argu->buff+extent*argu->low_seg_count*i, argu->low_seg_count, argu->dtype, argu->root_sm_rank, argu->low_comm, false);
            /* init the low task */
            init_task(low, mca_coll_future_bcast, (void *)(low_task_argu));
            /* add the low task into future, low task will be triggered by the future */
            add_tornado(low, f);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task %p future %p\n", argu->w_rank, (void *)low, (void *)f));
        }
        
        /* create bcast for next union segment if needed */
        if (argu->num_segments > argu->cur_seg + 1) {
            mca_coll_task_t *next = OBJ_NEW(mca_coll_task_t);
            /* set up next task arguments */ //for now, root has to 0//
            mca_bcast_next_argu_t *next_task_argu = malloc(sizeof(mca_bcast_next_argu_t));
            mac_coll_future_set_nextbcast_argu(next_task_argu, (char *)argu->buff+extent*max_seg_count, argu->up_seg_count, argu->low_seg_count, argu->dtype, argu->root_sm_rank, argu->root_leader_rank, argu->up_comm, argu->low_comm, argu->num_segments, argu->sm_rank, argu->cur_seg + 1, argu->w_rank, argu->last_seg_count);
            /* init the next task */
            init_task(next, mca_coll_future_nextbcast, (void *)(next_task_argu));
            /* add the nextbcast task into future, nextbcast task will be triggered by the future */
            add_tornado(next, f);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task %p future %p\n", argu->w_rank, (void *)next, (void *)f));
        }

        for (i=0; i<up_num; i++) {
            execute_task(up_list[i]);
        }

        free(up_list);
    }
    
    return OMPI_SUCCESS;
}

/* Build multilevel bcast based on non_blocking bcast
 * | IB0 IB1 | IB3 IB4 |
 * |    B2   |    B5   |
 * First task is issue IB0 and IB1 and wait all of them to finish.
 * First task will trigger mid task which will:
 * IB3
 * IB4
 * B5
 * Waitall(IB3, IB4)
 * to overlap IB3, IB4 and B2.
 */

int
mca_coll_future_bcast_intra_adapt(void *buff,
                                  int count,
                                  struct ompi_datatype_t *dtype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module)
{
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int w_size, w_rank, i;
    w_size = ompi_comm_size(comm);
    w_rank = ompi_comm_rank(comm);
    int up_seg_count = 65536;
    int low_seg_count = 524288;
    mca_coll_future_reset_seg_count(&up_seg_count, &low_seg_count, &count);
    int max_seg_count = (up_seg_count > low_seg_count) ? up_seg_count : low_seg_count;
    int up_num = max_seg_count / up_seg_count;
    int low_num = max_seg_count / low_seg_count;
    if (up_num > MAX_TASK_NUM || low_num > MAX_TASK_NUM) {
        return OMPI_ERROR;
    }
    int num_segments = (count + max_seg_count - 1) / max_seg_count;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "In Future up_count %d low_count %d count %d num_seg %d\n", up_seg_count, low_seg_count, count, num_segments));

    ompi_communicator_t *sm_comm;
    ompi_communicator_t *leader_comm;
    int *vranks;
    int sm_rank, sm_size;
    int leader_rank, leader_size;
    mca_coll_future_module_t *future_module = (mca_coll_future_module_t *)module;
    /* use cached communicators if possible */
    if (future_module->cached_comm == comm && future_module->cached_sm_comm != NULL && future_module->cached_leader_comm != NULL && future_module->cached_vranks != NULL) {
        sm_comm = future_module->cached_sm_comm;
        leader_comm = future_module->cached_leader_comm;
        vranks = future_module->cached_vranks;
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);
        leader_size = ompi_comm_size(leader_comm);
        leader_rank = ompi_comm_rank(leader_comm);
    }
    /* create communicators if there is no cached communicator */
    else {
        /* create sm_comm which contain all the process on a node */
        int var_id;
        int tmp_priority = 60;
        const int *origin_priority = NULL;
        int tmp_origin = 0;
        //const int *tmp = NULL;
        mca_base_var_find_by_name("coll_sm_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] sm_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("sm_priority after set %d %d\n", *tmp);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *)(&ompi_mpi_info_null), &sm_comm);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] sm_priority set back %d\n", w_rank, *tmp);
        sm_size = ompi_comm_size(sm_comm);
        sm_rank = ompi_comm_rank(sm_comm);
        
        /* create leader_comm which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_adapt_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d] adapt_priority origin %d %d\n", w_rank, *origin_priority, tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("adapt_priority after set %d %d\n", *tmp);
        ompi_comm_split(comm, sm_rank, w_rank, &leader_comm, false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        //mca_base_var_get_value(var_id, &tmp, NULL, NULL);
        //printf("[%d] adapt_priority set back %d\n", w_rank, *tmp);
        leader_size = ompi_comm_size(leader_comm);
        leader_rank = ompi_comm_rank(leader_comm);
        
        vranks = malloc(sizeof(int) * w_size);
        /* do allgather to gather vrank from each process so every process will know other processes vrank*/
        int vrank = sm_size * leader_rank + sm_rank;
        comm->c_coll->coll_allgather(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm, comm->c_coll->coll_allgather_module);
        future_module->cached_comm = comm;
        future_module->cached_sm_comm = sm_comm;
        future_module->cached_leader_comm = leader_comm;
        future_module->cached_vranks = vranks;
    }
    
    int root_sm_rank;
    int root_leader_rank;
    mca_coll_future_get_ranks(vranks, root, sm_size, &root_sm_rank, &root_leader_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: root_sm_rank %d root_leader_rank %d\n", w_rank, root_sm_rank, root_leader_rank));
    
    /* create future */
    mca_coll_future_t *f = OBJ_NEW(mca_coll_future_t);
    
    /* create first tasks for the first union segment */
    mca_coll_task_t *first = OBJ_NEW(mca_coll_task_t);
    /* setup up first task arguments */
    mca_bcast_first_argu_t *first_argu = malloc(sizeof(mca_bcast_first_argu_t));
    mac_coll_future_set_first_argu(first_argu, (char *)buff, up_seg_count, dtype, root_leader_rank, leader_comm, up_num, sm_rank!=root_sm_rank);
    /* init the first task */
    init_task(first, mca_coll_future_first_task, (void *)(first_argu));
    /* add the first task into future, up task will trigger the future */
    add_butterfly(first, f);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add butterfly task first %p future %p\n", w_rank, (void *)first, (void *)f));

    /* create mid task if needed */
    if (num_segments > 1) {
        mca_coll_task_t *mid = OBJ_NEW(mca_coll_task_t);
        /* set up mid task arguments */
        mca_bcast_mid_argu_t *mid_argu = malloc(sizeof(mca_bcast_mid_argu_t));
        mac_coll_future_set_mid_argu(mid_argu, (char *)buff, up_seg_count, low_seg_count, dtype, root_sm_rank, root_leader_rank, leader_comm, sm_comm, up_num, low_num, num_segments, 1, w_rank, count-(num_segments-1)*max_seg_count, sm_rank!=root_sm_rank);
        /* init the next task */
        init_task(mid, mca_coll_future_mid_task, (void *)(mid_argu));
        /* add the nextbcast task into future, nextbcast task will be triggered by the future */
        add_tornado(mid, f);
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task mid %p future %p\n", w_rank, (void *)mid, (void *)f));
    }
    else {
        /* create last tasks for the last union segment */
        for (i=0; i<low_num; i++) {
            mca_coll_task_t *last = OBJ_NEW(mca_coll_task_t);
            /* set up last task arguments */
            mca_bcast_argu_t *last_argu = malloc(sizeof(mca_bcast_argu_t));
            mac_coll_future_set_bcast_argu(last_argu, (char *)buff+extent*low_seg_count*i, low_seg_count, dtype, root_sm_rank, sm_comm, false);
            /* init the last task */
            init_task(last, mca_coll_future_bcast, (void *)(last_argu));
            /* add the last task into future, last task will be triggered by the future */
            add_tornado(last, f);
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: add tornado task last %p future %p\n", w_rank, (void *)last, (void *)f));
        }
    }
    
    execute_task(first);
    
    return OMPI_SUCCESS;
}

/* first task: issue and wait for the upper level ibcast of first union segment */
int mca_coll_future_first_task(void *task_argu){
    mca_bcast_first_argu_t *t = (mca_bcast_first_argu_t *)task_argu;
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    else {
        int i;
        ptrdiff_t extent, lb;
        ompi_datatype_get_extent(t->dtype, &lb, &extent);
        ompi_request_t **reqs = malloc(sizeof(ompi_request_t *)*t->num);
        for (i=0; i<t->num; i++) {
            t->comm->c_coll->coll_ibcast((char *)t->buff+extent*t->count*i, t->count, t->dtype, t->root, t->comm, &(reqs[i]), t->comm->c_coll->coll_ibcast_module);
        }
        ompi_request_wait_all(t->num, reqs, MPI_STATUSES_IGNORE);
        free(reqs);
        return OMPI_SUCCESS;
    }
}

/* mid task:
 *  1) issue upper level ibcast of next union segment
 *  2) issue and wait for lower level bcast of current union segment
 *  3) wait for the upper level ibcast of the next union segment to arrive
 */
int mca_coll_future_mid_task(void *task_argu){
    int i;
    mca_bcast_mid_argu_t *t = (mca_bcast_mid_argu_t *)task_argu;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: in mid\n", t->w_rank));
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    int max_seg_count = (t->up_seg_count > t->low_seg_count) ? t->up_seg_count : t->low_seg_count;
    /* no upper level ibcast needed */
    if (t->noop) {
        for (i=0; i<t->low_num; i++) {
            t->low_comm->c_coll->coll_bcast((char *)t->buff+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_sm_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
        }
    }
    else {
        if (t->num_segments == t->cur_seg + 1 && t->last_seg_count != max_seg_count) {
            ompi_request_t *req;
            t->up_comm->c_coll->coll_ibcast((char *)t->buff+extent*max_seg_count, t->last_seg_count, t->dtype, t->root_leader_rank, t->up_comm, &req, t->up_comm->c_coll->coll_ibcast_module);
            for (i=0; i<t->low_num; i++) {
                t->low_comm->c_coll->coll_bcast((char *)t->buff+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_sm_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
            }
            ompi_request_wait(&req, MPI_STATUSES_IGNORE);
        }
        else {
            ompi_request_t **reqs = malloc(sizeof(ompi_request_t *)*t->up_num);
            for (i=0; i<t->up_num; i++) {
                t->up_comm->c_coll->coll_ibcast((char *)t->buff+extent*max_seg_count+extent*t->up_seg_count*i, t->up_seg_count, t->dtype, t->root_leader_rank, t->up_comm, &(reqs[i]), t->up_comm->c_coll->coll_ibcast_module);
            }
            //ompi_request_wait_all(t->up_num, reqs, MPI_STATUSES_IGNORE);
            for (i=0; i<t->low_num; i++) {
                t->low_comm->c_coll->coll_bcast((char *)t->buff+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_sm_rank, t->low_comm, t->low_comm->c_coll->coll_bcast_module);
            }
            ompi_request_wait_all(t->up_num, reqs, MPI_STATUSES_IGNORE);
            free(reqs);
        }
    }
    /* create mid task for next union segment if needed */
    if (t->num_segments > t->cur_seg + 1) {
        mca_coll_task_t *mid = OBJ_NEW(mca_coll_task_t);
        /* set up mid task arguments */
        mca_bcast_mid_argu_t *mid_argu = malloc(sizeof(mca_bcast_mid_argu_t));
        mac_coll_future_set_mid_argu(mid_argu, (char *)t->buff+extent*max_seg_count, t->up_seg_count, t->low_seg_count, t->dtype,t->root_sm_rank, t->root_leader_rank, t->up_comm, t->low_comm, t->up_num, t->low_num, t->num_segments, t->cur_seg+1, t->w_rank, t->last_seg_count, t->noop);
        /* init the next task */
        init_task(mid, mca_coll_future_mid_task, (void *)(mid_argu));
        /* add the nextbcast task into future, nextbcast task will be triggered by the future */
        OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: execute mid %p\n", t->w_rank, (void *)mid));
        execute_task(mid);
    }
    /* create last tasks for the last union segment */
    else {
        if (t->last_seg_count != max_seg_count) {
            mca_coll_task_t *last = OBJ_NEW(mca_coll_task_t);
            /* set up last task arguments */
            mca_bcast_argu_t *last_argu = malloc(sizeof(mca_bcast_argu_t));
            mac_coll_future_set_bcast_argu(last_argu, (char *)t->buff+extent*max_seg_count, t->last_seg_count, t->dtype, t->root_sm_rank, t->low_comm, false);
            /* init the last task */
            init_task(last, mca_coll_future_bcast, (void *)(last_argu));
            /* add the last task into future, last task will be triggered by the future */
            OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: execute last0 %p\n", t->w_rank, (void *)last));
            execute_task(last);
        } else {
            for (i=0; i<t->low_num; i++) {
                mca_coll_task_t *last = OBJ_NEW(mca_coll_task_t);
                /* set up last task arguments */
                mca_bcast_argu_t *last_argu = malloc(sizeof(mca_bcast_argu_t));
                mac_coll_future_set_bcast_argu(last_argu, (char *)t->buff+extent*max_seg_count+extent*t->low_seg_count*i, t->low_seg_count, t->dtype, t->root_sm_rank, t->low_comm, false);
                /* init the last task */
                init_task(last, mca_coll_future_bcast, (void *)(last_argu));
                /* add the last task into future, last task will be triggered by the future */
                OPAL_OUTPUT_VERBOSE((30, mca_coll_future_component.future_output, "[%d]: execute last1 %p\n", t->w_rank, (void *)last));
                execute_task(last);
            }
        }
        
    }

    return OMPI_SUCCESS;
}




