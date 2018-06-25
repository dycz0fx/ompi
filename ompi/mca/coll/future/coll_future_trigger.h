#include "coll_future.h"

typedef int (*task_func_ptr) (void *);

struct mca_bcast_argu_s {
    void *buff;
    int count;
    struct ompi_datatype_t *dtype;
    int root;
    struct ompi_communicator_t *comm;
    mca_coll_base_module_t *module;
    bool noop;
};
typedef struct mca_bcast_argu_s mca_bcast_argu_t;

/* left task will trigger right task by future*/
struct mca_coll_future_s {
    opal_object_t  super;
    /* number of left tasks needed to trigger the right tasks */
    int count;
    /* a list to store every right task */
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

/* add left task to future */
int add_left(mca_coll_task_t *t, mca_coll_future_t *f);

/* add right task to future */
int add_right(mca_coll_task_t *t, mca_coll_future_t *f);

/* run the task */
int execute_task(mca_coll_task_t *t);

/* trigger the future */
int trigger_future(mca_coll_future_t *f);
