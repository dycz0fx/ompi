#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>


int32_t opal_ddt_generic_simple_pack_function_cuda_vector(opal_convertor_t* pConvertor,
                                                      struct iovec* iov,
                                                      uint32_t* out_size,
                                                      size_t* max_data )
{
    return 1;
}

int32_t opal_ddt_generic_simple_pack_function_cuda_vector2(opal_convertor_t* pConvertor,
                                                      struct iovec* iov,
                                                      uint32_t* out_size,
                                                      size_t* max_data )
{
    return 1;
}

void pack_contiguous_loop_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);


#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    printf("extent %ld, size %ld, count %ld\n", _loop->extent, _end_loop->size, _copy_loops);
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
    cudaMemcpy2DAsync(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#else
    pack_contiguous_loop_cuda_kernel_global<<<16, 8*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
}

/* this function will not be used */
void pack_contiguous_loop_cuda_pipeline( dt_elem_desc_t* ELEM,
                                         uint32_t* COUNT,
                                         unsigned char** SOURCE,
                                         unsigned char** DESTINATION,
                                         size_t* SPACE, unsigned char* gpu_buffer )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination_host = *(DESTINATION);
    unsigned char* _destination_dev = gpu_buffer;
    int i, pipeline_blocks;
    uint32_t _copy_loops_per_pipeline; 
    
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_pipeline\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)
 //   _source = pBaseBuf_GPU;
 //   _destination = (unsigned char*)cuda_desc_h->iov[0].iov_base;
#endif

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
 //   tasks_per_block = THREAD_PER_BLOCK * TASK_PER_THREAD;
 //   num_blocks = (*COUNT + tasks_per_block - 1) / tasks_per_block;
//    cudaMemcpy2D(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice);
    pipeline_blocks = 4;
    cuda_streams->current_stream_id = 0;
    _copy_loops_per_pipeline = (_copy_loops + pipeline_blocks -1 )/ pipeline_blocks;
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops_per_pipeline, _end_loop->size, _loop->extent, _source, _destination_dev);
    for (i = 1; i <= pipeline_blocks; i++) {
        cudaMemcpyAsync(_destination_host, _destination_dev, _end_loop->size * _copy_loops_per_pipeline, cudaMemcpyDeviceToHost, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
        cuda_streams->current_stream_id ++;
        cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
        _source += _loop->extent * _copy_loops_per_pipeline;
        _destination_dev += _end_loop->size * _copy_loops_per_pipeline;
        _destination_host += _end_loop->size * _copy_loops_per_pipeline;
        if (i == pipeline_blocks) {
            _copy_loops_per_pipeline = _copy_loops - _copy_loops_per_pipeline * (pipeline_blocks - 1);
        }
        pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops_per_pipeline, _end_loop->size, _loop->extent, _source, _destination_dev);
    }
    cudaMemcpyAsync(_destination_host, _destination_dev, _end_loop->size * _copy_loops_per_pipeline, cudaMemcpyDeviceToHost, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaDeviceSynchronize();
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
} 

void pack_contiguous_loop_cuda_memcpy2d_d2h( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_memcpy2d\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    

    cudaMemcpy2DAsync(_destination, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToHost, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing with memcpy2d in %ld microsec\n", total_time ); );
#endif
}

void pack_contiguous_loop_cuda_zerocopy( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _source = (*SOURCE) + _end_loop->first_elem_disp;
    uint32_t _copy_loops = *(COUNT);
    uint32_t num_blocks, tasks_per_block;
    unsigned char* _destination = *(DESTINATION);
    unsigned char* _destination_dev;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time;
#endif

    DT_CUDA_DEBUG( opal_cuda_output( 2, "Pack using contiguous_loop_cuda_zerocopy\n"); );

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (uint32_t)(*(SPACE) / _end_loop->size);


#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    

    cudaError_t reg_rv = cudaHostGetDevicePointer((void **)&_destination_dev, (void *) _destination, 0);
    if (reg_rv != cudaSuccess) {
        const char *cuda_err = cudaGetErrorString(reg_rv);
        printf("can not get dev  mem, %s\n", cuda_err);
    }
#if OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL
    cudaMemcpy2DAsync(_destination_dev, _end_loop->size, _source, _loop->extent, _end_loop->size, _copy_loops, cudaMemcpyDeviceToDevice, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
#else
    pack_contiguous_loop_cuda_kernel_global<<<192, 4*THREAD_PER_BLOCK, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_loops, _end_loop->size, _loop->extent, _source, _destination_dev);
#endif /* OPAL_DATATYPE_VECTOR_USE_MEMCPY2D_AS_KERNEL */

#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)    
    *(SOURCE) = _source +  _loop->extent*_copy_loops - _end_loop->first_elem_disp;
    *(DESTINATION) = *(DESTINATION)  + _copy_loops * _end_loop->size;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
#endif
    
    cudaStreamSynchronize(cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]);
    
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG( opal_cuda_output( 2, "[Timing]: vector packing in %ld microsec\n", total_time ); );
#endif
}

int32_t opal_ddt_generic_simple_pack_function_cuda_iov( opal_convertor_t* pConvertor,
                                                        struct iovec* iov,
                                                        uint32_t* out_size,
                                                        size_t* max_data )
{      
    return 1; 
}

#if 0

int32_t opal_ddt_generic_simple_pack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor,
                                                                   struct iovec* iov,
                                                                   uint32_t* out_size,
                                                                   size_t* max_data )
{
    uint32_t i, j;
    uint32_t count_desc, nb_blocks_per_description, residue_desc;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    size_t length, buffer_size, length_per_iovec, dst_offset;
    unsigned char *destination, *destination_base;
    size_t total_packed, total_converted;
    int32_t complete_flag = 0;
    uint8_t buffer_isfull = 0, transfer_required, free_required;
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
    
    /*description = pConvertor->use_desc->desc;
    pStack = pConvertor->pStack + pConvertor->stack_pos;
    pElem = &(description[pStack->index]);
    printf("size elem %lu, size %d\n", pElem->elem.common.type, opal_datatype_basicDatatypes[pElem->elem.common.type]->size);
    */
    
//    assert(opal_datatype_basicDatatypes[pElem->elem.common.type]->size != 0);

 //   printf("buffer size %d, max_data %d\n", iov[0].iov_len, *max_data);
    if ((iov[0].iov_base == NULL) || opal_ddt_cuda_is_gpu_buffer(iov[0].iov_base)) {
        if (iov[0].iov_len == 0) {
            buffer_size = DT_CUDA_BUFFER_SIZE;
        } else {
            buffer_size = iov[0].iov_len;
        }
        
        if (iov[0].iov_base == NULL) {
            iov[0].iov_base = (unsigned char *)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            destination = (unsigned char *)iov[0].iov_base;
            pConvertor->gpu_buffer_ptr = destination;
            free_required = 1;
        } else {
            destination = (unsigned char *)iov[0].iov_base;
            free_required = 0;
        }
        transfer_required = 0;
    } else {
        buffer_size = iov[0].iov_len;
        if (OPAL_DATATYPE_VECTOR_USE_ZEROCPY) {
            pConvertor->gpu_buffer_ptr = NULL;
            transfer_required = 0;
            free_required = 0;
            cudaHostGetDevicePointer((void **)&destination, (void *)iov[0].iov_base, 0);
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_ddt_cuda_malloc_gpu_buffer(buffer_size, 0);
            }
            transfer_required = 1;
            free_required = 1;
            destination = pConvertor->gpu_buffer_ptr;
        }
    }   
    
    destination_base = destination;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack using IOV non cached, GPU base %p, pack to buffer %p\n", pConvertor->pBaseBuf, destination););

    cuda_iov_count = 1000;//CUDA_NB_IOV;
    total_packed = 0;
    total_converted = pConvertor->bConverted;
    cuda_streams->current_stream_id = 0;
    convertor_flags = pConvertor->flags;
  //  orig_stack_index = pStack->index;

#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
    DT_CUDA_DEBUG ( opal_cuda_output(4, "Pack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
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

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        for (i = 0; i < cuda_iov_count; i++) {
          /*  pElem = &(description[orig_stack_index+i]);*/
            if (buffer_size >= cuda_iov[i].iov_len) {
                length_per_iovec = cuda_iov[i].iov_len;
            } else {
                /*orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                length_per_iovec = buffer_size / orig_alignment * orig_alignment;
                buffer_isfull = 1;
            }
            buffer_size -= length_per_iovec;
            total_packed += length_per_iovec;
            
            /* check alignment */
            if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_DOUBLE == 0 && (uintptr_t)destination % ALIGNMENT_DOUBLE == 0 && length_per_iovec >= ALIGNMENT_DOUBLE) {
                alignment = ALIGNMENT_DOUBLE;
            } else if ((uintptr_t)(cuda_iov[i].iov_base) % ALIGNMENT_FLOAT == 0 && (uintptr_t)destination % ALIGNMENT_FLOAT == 0 && length_per_iovec >= ALIGNMENT_FLOAT) {
                alignment = ALIGNMENT_FLOAT;
            } else {
                alignment = ALIGNMENT_CHAR;
            }

            count_desc = length_per_iovec / alignment;
            residue_desc = length_per_iovec % alignment;
            nb_blocks_per_description = (count_desc + thread_per_block - 1) / thread_per_block;
            DT_CUDA_DEBUG ( opal_cuda_output(10, "Pack description %d, size %d, residue %d, alignment %d, nb_block_aquired %d\n", i, count_desc, residue_desc, alignment, nb_blocks_per_description); );
            for (j = 0; j < nb_blocks_per_description; j++) {
                cuda_iov_dist_h_current[nb_blocks_used].src = (unsigned char *)(cuda_iov[i].iov_base) + j * thread_per_block * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].dst = destination;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = alignment;
                if ( (j+1) * thread_per_block <= count_desc) {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = thread_per_block;
                } else {
                    cuda_iov_dist_h_current[nb_blocks_used].nb_elements = count_desc - j*thread_per_block; 
                }
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Pack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
                assert (nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
            }
            
            /* handle residue */
            if (residue_desc != 0) {
                /*orig_alignment = opal_datatype_basicDatatypes[pElem->elem.common.type]->size;*/
                orig_alignment = ALIGNMENT_CHAR;
                cuda_iov_dist_h_current[nb_blocks_used].src = (unsigned char *)(cuda_iov[i].iov_base) + length_per_iovec / alignment * alignment;
                cuda_iov_dist_h_current[nb_blocks_used].dst = destination;
                cuda_iov_dist_h_current[nb_blocks_used].element_alignment = orig_alignment;
                cuda_iov_dist_h_current[nb_blocks_used].nb_elements = (length_per_iovec - length_per_iovec / alignment * alignment) / orig_alignment;
#if defined (OPAL_DATATYPE_CUDA_DEBUG)
                assert(cuda_iov_dist_h_current[nb_blocks_used].nb_elements > 0);
#endif /* OPAL_DATATYPE_CUDA_DEBUG */
                destination += cuda_iov_dist_h_current[nb_blocks_used].nb_elements * orig_alignment;
                DT_CUDA_DEBUG( opal_cuda_output(12, "Pack \tblock %d, src %p, dst %p, nb_elements %d, alignment %d\n", nb_blocks_used, cuda_iov_dist_h_current[nb_blocks_used].src, cuda_iov_dist_h_current[nb_blocks_used].dst, cuda_iov_dist_h_current[nb_blocks_used].nb_elements, cuda_iov_dist_h_current[nb_blocks_used].element_alignment); );
                nb_blocks_used ++;
                assert (nb_blocks_used < CUDA_MAX_NB_BLOCKS*CUDA_IOV_MAX_TASK_PER_BLOCK);
            }
            
            if (buffer_isfull) {
                break;
            }
        }

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: Pack to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", destination_base, total_time,  cuda_iov_pipeline_block->cuda_stream_id, nb_blocks_used); );
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_non_cached_t)*(nb_blocks_used), cudaMemcpyHostToDevice, *cuda_stream_iov);
        opal_generic_simple_pack_cuda_iov_non_cached_kernel<<<nb_blocks, thread_per_block, 0, *cuda_stream_iov>>>(cuda_iov_dist_d_current, nb_blocks_used);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block->cuda_event, *cuda_stream_iov);
        opal_cuda_check_error(cuda_err);
        iov_pipeline_block_id ++;
        iov_pipeline_block_id = iov_pipeline_block_id % NB_STREAMS;
        
        /* buffer is full */
        if (buffer_isfull) {
            size_t total_converted_tmp = total_converted;
            pConvertor->flags = convertor_flags;
            total_converted += total_packed;
            opal_convertor_set_position_nocheck(pConvertor, &total_converted);
            total_packed = total_converted - total_converted_tmp;
            break;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        convertor_flags = pConvertor->flags;
//        orig_stack_index = pStack->index;
        complete_flag = opal_convertor_raw( pConvertor, cuda_iov, &cuda_iov_count, &length );
        DT_CUDA_DEBUG ( opal_cuda_output(4, "Pack complete flag %d, iov count %d, length %d, submit to CUDA stream %d\n", complete_flag, cuda_iov_count, length, cuda_streams->current_stream_id); );
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: ddt to iov in %ld microsec\n", total_time ); );
#endif
    }
    

    for (i = 0; i < NB_STREAMS; i++) {
        cudaStreamSynchronize(cuda_streams->opal_cuda_stream[i]);
    }
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (transfer_required) {
        cudaMemcpy(iov[0].iov_base, pConvertor->gpu_buffer_ptr, total_packed, cudaMemcpyDeviceToHost);
    } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d\n", move_time, transfer_required ); );
#endif

    iov[0].iov_len = total_packed;
    *max_data = total_packed;
    *out_size = 1;
    DT_CUDA_DEBUG ( opal_cuda_output(2, "Pack total packed %d\n", total_packed); );
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    DT_CUDA_DEBUG ( opal_cuda_output(2, "[Timing]: total packing in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ); );
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


int32_t opal_ddt_generic_simple_pack_function_cuda_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{       
    return OPAL_SUCCESS;
}

int32_t opal_ddt_generic_simple_pack_function_cuda_iov_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{  
    return OPAL_SUCCESS;
}

void pack_predefined_data_cuda( dt_elem_desc_t* ELEM,
                                uint32_t* COUNT,
                                unsigned char** SOURCE,
                                unsigned char** DESTINATION,
                                size_t* SPACE )
{
    uint32_t _copy_count = *(COUNT);
    size_t _copy_blength;
    ddt_elem_desc_t* _elem = &((ELEM)->elem);
    unsigned char* _source = (*SOURCE) + _elem->disp;
    uint32_t nb_blocks, tasks_per_block, thread_per_block;
    unsigned char* _destination = *(DESTINATION);
    
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
    
    pack_contiguous_loop_cuda_kernel_global<<<nb_blocks, thread_per_block, 0, cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id]>>>(_copy_count, _copy_blength, _elem->extent, _source, _destination);
    cuda_streams->current_stream_id ++;
    cuda_streams->current_stream_id = cuda_streams->current_stream_id % NB_STREAMS;
    
#if !defined(OPAL_DATATYPE_CUDA_DRY_RUN)  
    _copy_blength *= _copy_count;
    *(SOURCE)  = _source + _elem->extent*_copy_count - _elem->disp;
    *(DESTINATION) += _copy_blength;
    *(SPACE)  -= _copy_blength;
    *(COUNT)  -= _copy_count;
#endif
    
}

