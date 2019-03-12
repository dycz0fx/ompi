#include "opal/class/opal_list.h"
#include "coll_future_trigger.h"

struct mca_coll_future_item_s {
    opal_list_item_t super;
    mca_coll_task_t* task;  //have received from how many children
};

typedef struct mca_coll_future_item_s mca_coll_future_item_t;

OBJ_CLASS_DECLARATION(mca_coll_future_item_t);
