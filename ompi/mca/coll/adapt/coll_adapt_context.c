#include "coll_adapt_context.h"

static void mca_coll_adapt_bcast_context_constructor(mca_coll_adapt_bcast_context_t *bcast_context){
}

static void mca_coll_adapt_bcast_context_destructor(mca_coll_adapt_bcast_context_t *bcast_context){
}

OBJ_CLASS_INSTANCE(mca_coll_adapt_bcast_context_t, opal_free_list_item_t, mca_coll_adapt_bcast_context_constructor, mca_coll_adapt_bcast_context_destructor);