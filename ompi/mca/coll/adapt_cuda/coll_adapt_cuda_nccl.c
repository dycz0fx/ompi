#include "opal_config.h"

#include "coll_adapt_cuda_nccl.h"
#include "coll_adapt_cuda.h"
#include <dlfcn.h>
#include <unistd.h>

#include "opal/mca/installdirs/installdirs.h"

static coll_adapt_cuda_nccl_function_table_t coll_adapt_cuda_nccl_table;
static void *coll_adapt_cuda_nccl_handle = NULL;
static char *coll_adapt_cuda_nccl_lib = NULL;

#define COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN(handle, fname)            \
    do {                                                                            \
        char* _error;                                                               \
        *(void **)(&(coll_adapt_cuda_nccl_table.fname ## _p)) = dlsym((handle), # fname);    \
        if(NULL != (_error = dlerror()) )  {                                        \
            opal_output(0, "Finding %s error: %s\n", # fname, _error);              \
            coll_adapt_cuda_nccl_table.fname ## _p = NULL;                                   \
            return OPAL_ERROR;                                                      \
        }                                                                           \
    } while (0)

int coll_adapt_cuda_nccl_init(void)
{
    if (coll_adapt_cuda_nccl_handle == NULL) {
        if( NULL != coll_adapt_cuda_nccl_lib )
            free(coll_adapt_cuda_nccl_lib);
        asprintf(&coll_adapt_cuda_nccl_lib, "%s/%s", opal_install_dirs.libdir, "libnccl.so");

        coll_adapt_cuda_nccl_handle = dlopen(coll_adapt_cuda_nccl_lib , RTLD_LAZY);
        if (!coll_adapt_cuda_nccl_handle) {
            opal_output( 0, "Failed to load %s library: error %s\n", coll_adapt_cuda_nccl_lib, dlerror());
            coll_adapt_cuda_nccl_handle = NULL;
            return OPAL_ERROR;
        }
    
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_nccl_handle, ncclGetUniqueId );
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_nccl_handle, ncclCommInitRank );
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_nccl_handle, ncclCommDestroy );
        COLL_ADAPT_CUDA_FIND_NCCL_FUNCTION_OR_RETURN( coll_adapt_cuda_nccl_handle, ncclBcast );
    
        mca_coll_adapt_cuda_component.coll_adapt_cuda_use_nccl = 1;
        opal_output( 0, "coll_adapt_cuda_nccl_init done\n");
    }
    
    return 1;
}

int coll_adapt_cuda_nccl_fini(void)
{
    if (coll_adapt_cuda_nccl_handle != NULL) {
        coll_adapt_cuda_nccl_table.ncclGetUniqueId_p = NULL;
        coll_adapt_cuda_nccl_table.ncclCommInitRank_p = NULL;
        coll_adapt_cuda_nccl_table.ncclCommDestroy_p = NULL;
        coll_adapt_cuda_nccl_table.ncclBcast_p = NULL;
        
        dlclose(coll_adapt_cuda_nccl_handle);
        coll_adapt_cuda_nccl_handle = NULL;
        if( NULL != coll_adapt_cuda_nccl_lib ) {
            free(coll_adapt_cuda_nccl_lib);
        }
        coll_adapt_cuda_nccl_lib = NULL;
        mca_coll_adapt_cuda_component.coll_adapt_cuda_use_nccl = 0;
        opal_output( 0, "coll_adapt_cuda_nccl_fini done\n");
    }
    
    return 1;
}

int coll_adapt_cuda_nccl_get_unique_id(ncclUniqueId* uniqueId)
{
    if (coll_adapt_cuda_nccl_table.ncclGetUniqueId_p != NULL) {
        coll_adapt_cuda_nccl_table.ncclGetUniqueId_p(uniqueId);
        return 1;
    } else {
        opal_output(0, "ncclGetUniqueId function pointer is NULL\n");
        return -1;
    }
}

int coll_adapt_cuda_nccl_comm_init_rank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank)
{
    if (coll_adapt_cuda_nccl_table.ncclCommInitRank_p != NULL) {
        ncclResult_t ret = coll_adapt_cuda_nccl_table.ncclCommInitRank_p(comm, ndev, commId, rank);
        if (ret != ncclSuccess) {
            opal_output(0, "ncclCommInitRank error\n");
            return 0;
        }
        return 1;
    } else {
        opal_output(0, "ncclCommInitRank function pointer is NULL\n");
        return -1;
    }
}

int coll_adapt_cuda_nccl_comm_destroy(ncclComm_t comm)
{
    if (coll_adapt_cuda_nccl_table.ncclCommDestroy_p != NULL) {
        coll_adapt_cuda_nccl_table.ncclCommDestroy_p(comm);
        return 1;
    } else {
        opal_output(0, "ncclCommDestroy function pointer is NULL\n");
        return -1;
    }
}

int coll_adapt_cuda_nccl_bcast(void* buff, int count, ncclDataType_t datatype, int root, ncclComm_t comm, ncclStream_t stream)
{
    if (coll_adapt_cuda_nccl_table.ncclBcast_p != NULL) {
        ncclResult_t ret = coll_adapt_cuda_nccl_table.ncclBcast_p(buff, count, datatype, root, comm, stream);
        if (ret != ncclSuccess) {
            opal_output(0, "ncclCommInitRank error\n");
            return 0;
        }
        return 1;
    } else {
        opal_output(0, "ncclBcast function pointer is NULL\n");
        return -1;
    }
}

