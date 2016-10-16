#ifndef COLL_ADAPT_CUDA_NCCL_H
#define COLL_ADAPT_CUDA_NCCL_H

typedef struct ncclComm* ncclComm_t;
typedef void* ncclStream_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidDevicePointer    =  4,
               ncclInvalidRank             =  5,
               ncclUnsupportedDeviceCount  =  6,
               ncclDeviceNotFound          =  7,
               ncclInvalidDeviceIndex      =  8,
               ncclLibWrapperNotSet        =  9,
               ncclCudaMallocFailed        = 10,
               ncclRankMismatch            = 11,
               ncclInvalidArgument         = 12,
               ncclInvalidType             = 13,
               ncclInvalidOperation        = 14,
               nccl_NUM_RESULTS            = 15 } ncclResult_t;

typedef enum { ncclChar       = 0,
               ncclInt        = 1,
#ifdef CUDA_HAS_HALF
               ncclHalf       = 2,
#endif
               ncclFloat      = 3,
               ncclDouble     = 4,
               ncclInt64      = 5,
               ncclUint64     = 6,
               nccl_NUM_TYPES = 7 } ncclDataType_t;
                              
struct coll_adapt_cuda_nccl_function_table {
    ncclResult_t (*ncclGetUniqueId_p)(ncclUniqueId* uniqueId);
    ncclResult_t (*ncclCommInitRank_p)(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);    
    void (*ncclCommDestroy_p)(ncclComm_t comm);
    ncclResult_t (*ncclBcast_p)(void* buff, int count, ncclDataType_t datatype, int root, ncclComm_t comm, void* stream);
};

typedef struct coll_adapt_cuda_nccl_function_table coll_adapt_cuda_nccl_function_table_t;

int coll_adapt_cuda_nccl_init(void);

int coll_adapt_cuda_nccl_fini(void);

int coll_adapt_cuda_nccl_get_unique_id(ncclUniqueId* uniqueId);

int coll_adapt_cuda_nccl_comm_init_rank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);

int coll_adapt_cuda_nccl_comm_destroy(ncclComm_t comm);

int coll_adapt_cuda_nccl_bcast(void* buff, int count, ncclDataType_t datatype, int root, ncclComm_t comm, ncclStream_t stream);

#endif