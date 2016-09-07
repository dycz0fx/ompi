#include "opal/class/opal_list.h"

struct mca_coll_adapt_cuda_item_s {
    opal_list_item_t super;
    int id;     //fragment id
    int count;  //have received from how many children
};

typedef struct mca_coll_adapt_cuda_item_s mca_coll_adapt_cuda_item_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_item_t);