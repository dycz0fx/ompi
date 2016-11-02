#include "opal/class/opal_free_list.h"      //free list

struct mca_coll_adapt_cuda_inbuf_s {
    opal_free_list_item_t super;
    char *buff;
};

typedef struct mca_coll_adapt_cuda_inbuf_s mca_coll_adapt_cuda_inbuf_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_inbuf_t);