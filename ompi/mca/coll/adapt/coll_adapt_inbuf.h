/*
 * Copyright (c) 2014-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_ADAPT_INBUF_H
#define MCA_COLL_ADAPT_INBUF_H

#include "opal/class/opal_free_list.h"      //free list

struct mca_coll_adapt_inbuf_s {
    opal_free_list_item_t super;
    char buff[1];
};

typedef struct mca_coll_adapt_inbuf_s mca_coll_adapt_inbuf_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_inbuf_t);

mca_coll_adapt_inbuf_t* coll_adapt_alloc_inbuf(opal_free_list_t *inbuf_list, size_t size, int buff_type);

int coll_adapt_free_inbuf(opal_free_list_t *inbuf_list, mca_coll_adapt_inbuf_t *inbuf, int buff_type);

#endif /* MCA_COLL_ADAPT_INBUF_H */
