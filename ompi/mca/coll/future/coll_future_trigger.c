#include "coll_future_trigger.h"

static void mca_coll_future_constructor(mca_coll_future_t *f){
    printf("create_future\n");
    f->count = 0;
    f->task_list = (mca_coll_task_t **)malloc(sizeof(mca_coll_task_t *) * 2);
    f->task_list_size = 0;
}

static void mca_coll_future_destructor(mca_coll_future_t *f){
    free(f->task_list);
}

OBJ_CLASS_INSTANCE(mca_coll_future_t, opal_object_t, mca_coll_future_constructor, mca_coll_future_destructor);

static void mca_coll_task_constructor(mca_coll_task_t *t){
    printf("create_task\n");
    t->future_list = (mca_coll_future_t **)malloc(sizeof(mca_coll_future_t *) * 2);
    t->future_list_size = 0;
}

static void mca_coll_task_destructor(mca_coll_task_t *t){
    free(t->future_list);
}

OBJ_CLASS_INSTANCE(mca_coll_task_t, opal_object_t, mca_coll_task_constructor, mca_coll_task_destructor);

/* init future */
int init_future(mca_coll_future_t *f){
    return OMPI_SUCCESS;
}

/* init task */
int init_task(mca_coll_task_t *t, task_func_ptr func_ptr, void *func_argu){
    t->func_ptr = func_ptr;
    t->func_argu = func_argu;
    return OMPI_SUCCESS;
}

/* add butterfly task to future */
int add_butterfly(mca_coll_task_t *t, mca_coll_future_t *f){
    printf("add_butterfly\n");
    f->count++;
    t->future_list[t->future_list_size] = f;
    t->future_list_size++;
    return OMPI_SUCCESS;
}

/* add tornado task to future */
int add_tornado(mca_coll_task_t *t, mca_coll_future_t *f){
    printf("add_tornado\n");
    f->task_list[f->task_list_size] = t;
    f->task_list_size++;
    return OMPI_SUCCESS;
}

/* run the task */
int execute_task(mca_coll_task_t *t){
    printf("execute_taks %p\n", (void *)t);
    t->func_ptr(t->func_argu);
    int i;
    for (i=0; i<t->future_list_size; i++) {
        trigger_future(t->future_list[i]);
    }
    return OMPI_SUCCESS;
}

/* trigger the future */
int trigger_future(mca_coll_future_t *f){
    printf("trigger_future %p\n", (void *)f);
    int i;
    f->count--;
    if (f->count == 0) {
        for (i=0; i<f->task_list_size; i++) {
            execute_task(f->task_list[i]);
        }
    }
    return OMPI_SUCCESS;
}
