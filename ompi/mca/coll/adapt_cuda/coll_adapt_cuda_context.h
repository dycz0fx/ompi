#include "ompi/mca/coll/coll.h"
#include "opal/class/opal_free_list.h"      //free list
#include "opal/class/opal_list.h"       //list
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_topo.h"  //ompi_coll_tree_t
#include "coll_adapt_cuda_inbuf.h"

#define CPU_BUFFER_MEMCPY_DONE  1
#define CPU_BUFFER_MEMCPY_NOT_DONE  0

#define COLL_ADAPT_CUDA_CONTEXT_FLAGS_BCAST     0x2
#define COLL_ADAPT_CUDA_CONTEXT_FLAGS_REDUCE     0x4

/* bcast constant context in bcast context */
struct mca_coll_adapt_cuda_constant_bcast_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int real_seg_size;
    int num_segs;
    ompi_request_t * request;
    opal_mutex_t * mutex;
    opal_free_list_t * context_list;
    int* recv_array;
    int* send_array;
    int num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int num_sent_segs;  //number of sent segments
    ompi_coll_tree_t * tree;
    char **cpu_buff_list;
    int *cpu_buff_memcpy_flags;
    int *cpu_buff_list_ref_count;
};

typedef struct mca_coll_adapt_cuda_constant_bcast_context_s mca_coll_adapt_cuda_constant_bcast_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_bcast_context_t);


//bcast context
typedef struct mca_coll_adapt_cuda_bcast_context_s mca_coll_adapt_cuda_bcast_context_t;

typedef int (*mca_coll_adapt_cuda_bcast_cuda_callback_fn_t)(mca_coll_adapt_cuda_bcast_context_t *context);

struct mca_coll_adapt_cuda_bcast_context_s {
    opal_free_list_item_t super;
    int flags;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    size_t send_count;
    mca_coll_adapt_cuda_constant_bcast_context_t * con;
    mca_coll_adapt_cuda_bcast_cuda_callback_fn_t cuda_callback; 
};

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_bcast_context_t);


/* ibcast constant context in ibcast context */
struct mca_coll_adapt_cuda_constant_ibcast_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int real_seg_size;
    int num_segs;
    opal_free_list_t * context_list;
    int* recv_array;
    int* send_array;
    int num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int num_sent_segs;  //number of sent segments
    opal_mutex_t * mutex;
    ompi_request_t * request;
    ompi_coll_tree_t * tree;
};

typedef struct mca_coll_adapt_cuda_constant_ibcast_context_s mca_coll_adapt_cuda_constant_ibcast_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_ibcast_context_t);


//ibcast context
struct mca_coll_adapt_cuda_ibcast_context_s {
    opal_free_list_item_t super;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    mca_coll_adapt_cuda_constant_ibcast_context_t * con;
};

typedef struct mca_coll_adapt_cuda_ibcast_context_s mca_coll_adapt_cuda_ibcast_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_ibcast_context_t);



/* reduce constant context in reduce context */
struct mca_coll_adapt_cuda_constant_reduce_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int segment_increment;      //increment of each segment
    int num_segs;
    ompi_request_t * request;
    int rank;      //change, unused
    opal_free_list_t * context_list;
    int32_t num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int32_t num_sent_segs;  //number of sent segments
    int32_t* next_recv_segs;  //next seg need to be received for every children
    opal_mutex_t * mutex;     //old, only for test
    opal_mutex_t * mutex_recv_list;     //use to lock recv list
    opal_mutex_t * mutex_num_recv_segs;     //use to lock num_recv_segs
    opal_mutex_t * mutex_num_sent;     //use to lock num_sent
    opal_mutex_t ** mutex_op_list;   //use to lock each segment when do the reduce op
    ompi_op_t * op;  //reduce operation
    ompi_coll_tree_t * tree;
    char ** accumbuf;   //accumulate buff, used in reduce
    opal_free_list_t *inbuf_list;
    opal_list_t *recv_list;    //a list to store the segments which are received and not yet be sent
    ptrdiff_t lower_bound;
    int32_t ongoing_send;   //how many send is posted but not finished
    char * sbuf;    //inputed sbuf
    char * rbuf;    //inputed rbuf
    int root;
    int distance;   //address of inbuf->buff to address of inbuf
    size_t real_seg_size;
    char **cpu_buff_list;
    char ** accumbuf_cpu;   //accumulate buff for cpu use, used in reduce
};

typedef struct mca_coll_adapt_cuda_constant_reduce_context_s mca_coll_adapt_cuda_constant_reduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_reduce_context_t);


//reduce context
typedef struct mca_coll_adapt_cuda_reduce_context_s mca_coll_adapt_cuda_reduce_context_t;

typedef int (*mca_coll_adapt_cuda_reduce_cuda_callback_fn_t)(mca_coll_adapt_cuda_reduce_context_t *context);

struct mca_coll_adapt_cuda_reduce_context_s {
    opal_free_list_item_t super;
    int flags;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    size_t count;
    mca_coll_adapt_cuda_constant_reduce_context_t * con;
    char *inbuf;  //only used in reduce, store the incoming segment
    void *buff_to_free_item;
    mca_coll_adapt_cuda_reduce_cuda_callback_fn_t cuda_callback; 
};

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_reduce_context_t);


/* ireduce constant context in ireduce context */
struct mca_coll_adapt_cuda_constant_ireduce_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int segment_increment;      //increment of each segment
    int num_segs;
    int rank;      //change, unused
    opal_free_list_t * context_list;
    int num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int num_sent_segs;  //number of sent segments
    int* next_recv_segs;  //next seg need to be received for every children
    opal_mutex_t * mutex_recv_list;     //use to lock recv list
    opal_mutex_t * mutex_num_recv_segs;     //use to lock num_recv_segs
    opal_mutex_t ** mutex_op_list;   //use to lock each segment when do the reduce op
    ompi_request_t * request;
    ompi_op_t * op;  //reduce operation
    ompi_coll_tree_t * tree;
    char * accumbuf;   //accumulate buff
    opal_free_list_t *inbuf_list;
    opal_list_t *recv_list;    //a list to store the segments which are received and not yet be sent
    ptrdiff_t lower_bound;
    int ongoing_send;   //how many send is posted but not finished
    char * accumbuf_free;   //use to free the accumbuf
    
};

typedef struct mca_coll_adapt_cuda_constant_ireduce_context_s mca_coll_adapt_cuda_constant_ireduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_ireduce_context_t);


//ireduce context
struct mca_coll_adapt_cuda_ireduce_context_s {
    opal_free_list_item_t super;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    mca_coll_adapt_cuda_constant_ireduce_context_t * con;
    mca_coll_adapt_cuda_inbuf_t *inbuf;  //store the incoming segmetn
};

typedef struct mca_coll_adapt_cuda_ireduce_context_s mca_coll_adapt_cuda_ireduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_ireduce_context_t);

/* bcast constant context in bcast context for two trees */
struct mca_coll_adapt_cuda_constant_bcast_two_trees_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int real_seg_size;
    int* num_segs;
    ompi_request_t * request;
    opal_free_list_t ** context_lists;
    int** recv_arrays;
    int** send_arrays;
    int *num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int *num_sent_segs;  //number of sent segments
    opal_mutex_t * mutex;
    ompi_coll_tree_t ** trees;
    int complete;
};

typedef struct mca_coll_adapt_cuda_constant_bcast_two_trees_context_s mca_coll_adapt_cuda_constant_bcast_two_trees_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_bcast_two_trees_context_t);


//bcast context for two trees
struct mca_coll_adapt_cuda_bcast_two_trees_context_s {
    opal_free_list_item_t super;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    int tree; //which tree are using
    mca_coll_adapt_cuda_constant_bcast_two_trees_context_t * con;
};

typedef struct mca_coll_adapt_cuda_bcast_two_trees_context_s mca_coll_adapt_cuda_bcast_two_trees_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_bcast_two_trees_context_t);


/* bcast constant context in ibcast context for two trees */
struct mca_coll_adapt_cuda_constant_ibcast_two_trees_context_s {
    opal_object_t  super;
    size_t count;
    size_t seg_count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    int real_seg_size;
    int* num_segs;
    opal_free_list_t ** context_lists;
    int** recv_arrays;
    int** send_arrays;
    int *num_recv_segs; //store the length of the fragment array, how many fragments are recevied
    int *num_sent_segs;  //number of sent segments
    opal_mutex_t * mutex;
    ompi_request_t * request;
    ompi_coll_tree_t ** trees;
    int complete;
};

typedef struct mca_coll_adapt_cuda_constant_ibcast_two_trees_context_s mca_coll_adapt_cuda_constant_ibcast_two_trees_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_ibcast_two_trees_context_t);


//bcast context for two trees
struct mca_coll_adapt_cuda_ibcast_two_trees_context_s {
    opal_free_list_item_t super;
    char *buff;
    int frag_id;
    int child_id;
    int peer;
    int tree; //which tree are using
    mca_coll_adapt_cuda_constant_ibcast_two_trees_context_t * con;
};

typedef struct mca_coll_adapt_cuda_ibcast_two_trees_context_s mca_coll_adapt_cuda_ibcast_two_trees_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_ibcast_two_trees_context_t);


/* allreduce constant context in allreduce context */
struct mca_coll_adapt_cuda_constant_allreduce_context_s {
    opal_object_t  super;
    char *sendbuf;
    char *recvbuf;
    size_t count;
    ompi_datatype_t * datatype;
    ompi_communicator_t * comm;
    ompi_request_t * request;
    opal_free_list_t * context_list;
    ompi_op_t * op;  //reduce operation
    ptrdiff_t lower_bound;
    int extra_ranks;
    opal_free_list_t *inbuf_list;
    int complete;
    int adjsize;
    int sendbuf_ready;
    int inbuf_ready;
};

typedef struct mca_coll_adapt_cuda_constant_allreduce_context_s mca_coll_adapt_cuda_constant_allreduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_allreduce_context_t);


//allreduce context
struct mca_coll_adapt_cuda_allreduce_context_s {
    opal_free_list_item_t super;
    mca_coll_adapt_cuda_inbuf_t *inbuf;  //store the incoming segment
    int newrank;
    int distance;      //distance for recursive doubleing
    int peer;
    mca_coll_adapt_cuda_constant_allreduce_context_t * con;
};

typedef struct mca_coll_adapt_cuda_allreduce_context_s mca_coll_adapt_cuda_allreduce_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_allreduce_context_t);

/* alltoallv constant context in alltoallv context */
struct mca_coll_adapt_cuda_constant_alltoallv_context_s {
    opal_object_t  super;
    char *sbuf;
    const int* scounts;
    const int* sdisps;
    ompi_datatype_t * sdtype;
    char *rbuf;
    const int* rcounts;
    const int* rdisps;
    ompi_datatype_t * rdtype;
    ompi_communicator_t * comm;
    ompi_request_t * request;
    opal_free_list_t * context_list;
    ptrdiff_t sext;
    ptrdiff_t rext;
    const char *origin_sbuf;
    int finished_send;
    int finished_recv;
    int ongoing_send;
    int ongoing_recv;
    int complete;
    int next_send_distance;
    int next_recv_distance;
};

typedef struct mca_coll_adapt_cuda_constant_alltoallv_context_s mca_coll_adapt_cuda_constant_alltoallv_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_constant_alltoallv_context_t);


//alltoallv context
struct mca_coll_adapt_cuda_alltoallv_context_s {
    opal_free_list_item_t super;
    int distance;      //distance for recursive doubleing
    int peer;
    char *start;         //point to send or recv from which address
    mca_coll_adapt_cuda_constant_alltoallv_context_t * con;
};

typedef struct mca_coll_adapt_cuda_alltoallv_context_s mca_coll_adapt_cuda_alltoallv_context_t;

OBJ_CLASS_DECLARATION(mca_coll_adapt_cuda_alltoallv_context_t);
