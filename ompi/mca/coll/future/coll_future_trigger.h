#include "coll_future.h"

typedef int (*task_func_ptr) (void *);

/* bufferfly task will trigger tornado task by future*/
struct mca_coll_future_s {
    opal_object_t  super;
    /* number of bufferfly tasks needed to trigger the tornado tasks */
    int count;
    /* a list to store every tornado task */
    struct mca_coll_task_s **task_list;
    int task_list_size;
};

struct mca_coll_task_s {
    opal_object_t  super;
    /* the list of futures it needs to trigger */
    struct mca_coll_future_s **future_list;
    int future_list_size;
    /* the functhion in the task */
    task_func_ptr func_ptr;
    void *func_argu;
};

typedef struct mca_coll_future_s mca_coll_future_t;
typedef struct mca_coll_task_s mca_coll_task_t;

OBJ_CLASS_DECLARATION(mca_coll_future_t);
OBJ_CLASS_DECLARATION(mca_coll_task_t);

/* init future */
int init_future(mca_coll_future_t *f);

/* init task */
int init_task(mca_coll_task_t *t, task_func_ptr func_ptr, void *func_argu);

/* add butterfly task to future */
int add_butterfly(mca_coll_task_t *t, mca_coll_future_t *f);

/* add tornado task to future */
int add_tornado(mca_coll_task_t *t, mca_coll_future_t *f);

/* run and complete task */
int execute_task(mca_coll_task_t *t);

/* issue the task, non blocking version of execute_task */
int issue_task(mca_coll_task_t *t);

/* complete the task, complete the corresponding task */
int complete_task(mca_coll_task_t *t);

/* trigger the future */
int trigger_future(mca_coll_future_t *f);

/* free the task */
void free_task(mca_coll_task_t *t);

/* free the future */
void free_future(mca_coll_future_t *f);
