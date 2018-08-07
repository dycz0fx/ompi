#include "coll_adapt.h"

/* bcast constant context in bcast context */
struct mca_coll_adapt_bcast_context_s {
    opal_free_list_item_t super;
    void *buff;
    int root;
    int count;
    ompi_datatype_t *datatype;
    ompi_communicator_t *comm;
    ompi_request_t *request;
    opal_mutex_t *mutex;
    ompi_coll_tree_t *tree;
    int ibcast_tag;
    int peer;
    int *num_sent;
    int *num_recv;
};

typedef struct mca_coll_adapt_bcast_context_s mca_coll_adapt_bcast_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_bcast_context_t);
