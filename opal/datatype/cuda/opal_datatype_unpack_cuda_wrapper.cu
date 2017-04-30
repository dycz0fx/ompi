#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>


int32_t opal_ddt_generic_simple_unpack_function_cuda_vector( opal_convertor_t* pConvertor,
                                                         struct iovec* iov, uint32_t* out_size,
                                                         size_t* max_data )
{
    return 1;
}

int32_t opal_ddt_generic_simple_unpack_function_cuda_vector2( opal_convertor_t* pConvertor,
                                                         struct iovec* iov, uint32_t* out_size,
                                                         size_t* max_data )
{
    return 0;
}


int32_t opal_ddt_generic_simple_unpack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                          struct iovec* iov,
                                                          uint32_t* out_size,
                                                          size_t* max_data )
{
    return 1;
}

#if 0
int32_t opal_ddt_generic_simple_unpack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor,
                                                                     struct iovec* iov,
                                                                     uint32_t* out_size,
                                                                     size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, nb_blocks_per_description, dst_offset, residue_desc;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    size_t length, buffer_size, length_per_iovec;
    unsigned char *source, *source_base;
    size_t total_unpacked, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0;
    uint8_t free_required = 0;
    uint32_t convertor_flags;
//    dt_elem_desc_t* description;
//    dt_elem_desc_t* pElem;
//    dt_stack_t* pStack;
    uint8_t alignment, orig_alignment;
//    int32_t orig_stack_index;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_non_cached_t* cuda_iov_dist_d_current;
    ddt_cuda_iov_pipeline_block_t *cuda_iov_pipeline_block;
    int iov_pipeline_block_id = 0;
    cudaStream_t *cuda_stream_iov = NULL;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif

/*    description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
    printf("size elem %d, size %lu\n", pElem->elem.common.type, opal_datatype_basicDatatypes[pElem->elem.common.type]->size);
*/

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (opal_ddt_cuda_is_gpu_buffer(iov[0].iov_base)) {
        source = (unsigned char*)iov[0].iov_base;
        free_required = 0;
    } else {
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            cudaHostGetDevicePointer((void **)&source, (void *)iov[0].iov_base, 0);
            pConvertor->gpu_buffer_ptr = NULL;
            free_required = 0;
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(iov[0].iov_len, 0);
            }
            source = pConvertor->gpu_buffer_ptr;
            cudaMemcpy(source, iov[0].iov_base, iov[0].iov_len, cudaMemcpyHostToDevice);
            free_required = 1;
        }
    }

    source_base = source;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack using IOV non cached, GPU base %p, unpack from buffer %p, total size %ld\n",
                                     pConvertor->pBaseBuf, source, iov[0].iov_len); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: HtoD memcpy in %ld microsec, free required %d\n", move_time, free_required ); );
#endif
    
//    cuda_err = cudaEventRecord(current_cuda_device->memcpy_event, current_cuda_device->cuda_streams->opal_cuda_stream[0]);
//    opal_cuda_check_error(cuda_err);


#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    buffer_size = iov[0].iov_len;
    cuda_iov_count = 1000;
    total_unpacked = 0;
    total_converted = pConvertor->bConverted;
    cuda_streams->current_stream_id = 0;
    convertor_flags = pConvertor->flags;
//    orig_stack_index = pStack->index;
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(4, "Unpack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined (OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif

    dst_offset = 0;
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;

    while (cuda_iov_count > 0) {

        nb_blocks_used = 0;
        cuda_iov_pipeline_block = current_cuda_device->cuda_iov_pipeline_block[iov_pipeline_block_id];
        cuda_iov_dist_h_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_h;
        cuda_iov_dist_d_current = cuda_iov_pipeline_block->cuda_iov_dist_non_cached_d;
        cuda_stream_iov = cuda_iov_pipeline_block->cuda_stream;
        cuda_err = cudaStreamWaitEvent(*cuda_stream_iov, cuda_iov_pipeline_block->cuda_event, 0);
        opal_cuda_check_error(cuda_err);
        

#if defined (OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        for (i = 0; i < cuda_iov_count; i++) {
//            pElem = &(description[orig_stack_index+i]);
            if (buffer_size >= cuda_iov[i].iov_len) {
                length_per_iovec = cuda_iov[i].iov_len;
            } else {
              /*  orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                length_per_iovec = buffer_size / orig_alignment * orig_alignment;
                buffer_isfull = 1;
            }
            buffer_size -= length_per_iovec;
            total_unpacked += length_per_iovec;

            /* check alignment */
            if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)source % ALIGNMENT_DOUBLE == 0 && length_per_iovec >= ALIGNMENT_DOUBLE) {
                alignment = ALIGNMENT_DOUBLE;
            } else if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_FLOAT == 0 && (uintptr_t)source % ALIGNMENT_FLOAT == 0 && length_per_iovec >= ALIGNMENT_FLOAT) {
                alignment = ALIGNMENT_FLOAT;
            } else {
                alignment = ALIGNMENT_CHAR;
            }

            //alignment = ALIGNMENT_DOUBLE;

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "Unpack description %d, size %d, residue %d, alignment %d\n", i, count_desc, residue_desc, alignment); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[nb_blocks_used].dst = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].src = source;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = alignment;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = thread_per_block;// * sizeof(double);
                } else {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (thread_per_block - ((j+1)*thread_per_block - count_desc));// * sizeof(double);
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0); 
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Unpack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
            }

            /* handle residue */
            if (residue_desc != 0) {
               /* orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                cuda_iov_dist_h_current[nb_blocks_used].dst = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].src = source;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = orig_alignment;
                cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert (cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                source += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Unpack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
            }

            if (buffer_isfull) {
                break;
            }
        }

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Unpack src %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks_used %d\n", source_base, total_time,  cuda_iov_pipeline_block->cuda_stream_id, nb_blocks_used); );
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_non_cached_t)*(nb_blocks_used), cudaMemcpyHostToDevice, *cuda_stream_iov);
        opal_generic_simple_unpack_cuda_iov_non_cached_kernel<<<nb_blocks, thread_per_block, 0, *cuda_stream_iov>>>(cuda_iov_dist_d_current, nb_blocks_used);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block->cuda_event, *cuda_stream_iov);
        opal_cuda_check_error(cuda_err);
        iov_pipeline_block_id ++;
        iov_pipeline_block_id = iov_pipeline_block_id % NB_STREAMS;
        
        /* buffer is full */
        if (buffer_isfull) {
            size_t total_converted_tmp = total_converted;
            pConvertor->flags = convertor_flags;
            total_converted += total_unpacked;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            total_unpacked = total_converted - total_converted_tmp;
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        convertor_flags = pConvertor->flags;
//        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(4, "Unpack complete flag %d, iov count %d, length %d, submit to CUDA stream %d, nb_blocks %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id, nb_blocks_used); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif

    }

    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }

    iov[0].iov_len = total_unpacked;
    *max_data = total_unpacked;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Unpack total unpacked %d\n", total_unpacked); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total unpacking in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
#endif

    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required) {
            opal_ddt_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
            pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }
    return 0;
}

#endif

int32_t opal_ddt_generic_simple_unpack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked)
{
    return OPAL_SUCCESS;
}

int32_t opal_ddt_generic_simple_unpack_function_cuda_iov_cached( opal_convertor_t* pConvertor, unsigned char *source, size_t buffer_size, size_t *total_unpacked)
{
    return OPAL_SUCCESS;
}

void unpack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
     cudaMemcpy2DAsync(_destination, _loop->extent, _source, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#else
     unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif

    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking in %ld microsec\n", total_time ); );
#endif
}

void unpack_contiguous_loop_cuda_memcpy2d( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda_memcpy2d\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    cudaMemcpy2DAsync(_destination, _loop->extent, _source, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyHostToDevice, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking with memcpy2d in %ld microsec\n", total_time ); );
#endif
}

void unpack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                           uint32_t* COUNT,
                                           unsigned char** SOURCE,
                                           unsigned char** DESTINATION,
                                           size_t* SPACE)
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _destination = (*DESTINATION) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _source = *(SOURCE);
    unsigned char* _source_dev;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif
    
    DT_CUDA_DEBUG( opal_cuda_output( 2, "Unpack using contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
//    tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
//    num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;

    cudaError_t reg_rv = cudaHostGetDevicePointer((void **)&_source_dev, (void *) _source, 0);
    if (reg_rv != cudaSuccess) {
        const char *cuda_err = cudaGetErrorString(reg_rv);
        printf("can not get dev mem, %s\n", cuda_err);
    }
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
    cudaMemcpy2DAsync(_destination, _loop->extent, _source_dev, _end_loop->size, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#else
    unpack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops, _end_loop->size, _loop->extent, _source_dev, _destination);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
    *(DESTINATION) = _destination + _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(SOURCE) = *(SOURCE)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif

    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
  //  cudaHostUnregister(_source);
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector unpacking in %ld microsec\n", total_time ); );
#endif
}

void unpack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                  uint32_t* COUNT,
                                  unsigned char** SOURCE,
                                  unsigned char** DESTINATION,
                                  size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _source = (*SOURCE);
    uint32_t nb_blocks, tasks_per_block, thread_per_block;
    unsigned char* _destination = *(DESTINATION) + _elem->disp;
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;

    _copy_blength = 8;//opal_datatype_basicDatatypes[_elem->common.type]->size;
    if( (_copy_count * _copy_blength) > *(SPACE) ) {
        _copy_count = (uint32_t)(*(SPACE) / _copy_blength);
        if( 0 == _copy_count ) return;  /* nothing to do */
    }
    
    
    if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE) {
        thread_per_block = CUDA_WARP_SIZE;
    } else if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE * 2) {
        thread_per_block = CUDA_WARP_SIZE * 2;
    } else if (*COUNT / TASK_PER_THREAD < CUDA_WARP_SIZE * 3) {
        thread_per_block = CUDA_WARP_SIZE * 3;
    } else {
        thread_per_block = CUDA_WARP_SIZE * 5;
    }
    tasks_per_block = thread_per_block * TASK_PER_THREAD;
    nb_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;

 //   DBGPRINT("num_blocks %d, thread %d\n", nb_blocks, tasks_per_block);
 //   DBGPRINT( "GPU pack 1. memcpy( %p, %p, %lu ) => space %lu\n", _destination, _source, (unsigned long)_copy_count, (unsigned long)(*(SPACE)) );
    
    unpack_contiguous_loop_cuda_kernel_global<<<nb_blocks, thread_per_block, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_count, _copy_blength, _elem->extent, _source, _destination);
    cuda_streams->current_stream_id ++;
    cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)  
    _copy_blength *= _copy_count;
    *(DESTINATION)  = _destination + _elem->extent*_copy_count - _elem->disp;
    *(SOURCE) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
#endif
    
}
