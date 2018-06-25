#ifndef MCA_COLL_ADAPT_INBUF_H
#define MCA_COLL_ADAPT_INBUF_H

#include "opal/class/opal_free_list.h"      //free list

struct mca_coll_adapt_inbuf_s {
    opal_free_list_item_t super;
    char buff[1];
    //char* buff;
};

typedef struct mca_coll_adapt_inbuf_s mca_coll_adapt_inbuf_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_inbuf_t);

#endif /* MCA_COLL_ADAPT_INBUF_H */
