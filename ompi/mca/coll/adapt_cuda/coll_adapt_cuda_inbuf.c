#include "coll_adapt_cuda_inbuf.h"

static void mca_coll_adapt_cuda_inbuf_constructor(mca_coll_adapt_cuda_inbuf_t *inbuf){
}

static void mca_coll_adapt_cuda_inbuf_destructor(mca_coll_adapt_cuda_inbuf_t *inbuf){
}

OBJ_CLASS_INSTANCE(mca_coll_adapt_cuda_inbuf_t, opal_free_list_item_t, mca_coll_adapt_cuda_inbuf_constructor, mca_coll_adapt_cuda_inbuf_destructor);
