#include "coll_future.h"

/* reset the bigger segment count of up_seg_count and low_seg_count such that it can be exactly divieded by smaller segment count */
void mca_coll_future_reset_seg_count(int *up_seg_count, int *low_seg_count, int *count) {
    if (*up_seg_count > *count) {
        *up_seg_count = *count;
    }
    if (*low_seg_count > *count) {
        *low_seg_count = *count;
    }
    if (*up_seg_count == *low_seg_count) {
        return;
    }
    else if (*up_seg_count > *low_seg_count) {
        if (*up_seg_count % *low_seg_count == 0) {
            return;
        }
        else {
            int t = *up_seg_count / *low_seg_count;
            *up_seg_count = *low_seg_count * t;
            return;
        }
    }
    else {
        if (*low_seg_count % *up_seg_count == 0) {
            return;
        }
        else {
            int t = *low_seg_count / *up_seg_count;
            *low_seg_count = *up_seg_count * t;
            return;
        }
    }
}

/* get root's low_rank and up_rank from vranks array */
void mca_coll_future_get_ranks(int *vranks, int root, int low_size, int *root_low_rank, int *root_up_rank){
    *root_up_rank = vranks[root] / low_size;
    *root_low_rank = vranks[root] % low_size;
}

uint32_t future_auto_tuned_get_n(uint32_t n){
    uint32_t avail[4] = {16, 32, 64, 128};
    uint32_t i;
    for (i=0; i<4; i++) {
        if (avail[i] >= n) {
            return i;
        }
    }
    return i-1;
}

uint32_t future_auto_tuned_get_c(uint32_t c){
    uint32_t avail[1] = {32};
    uint32_t i;
    for (i=0; i<1; i++) {
        if (avail[i] >= c) {
            return i;
        }
    }
    return i-1;
}

uint32_t future_auto_tuned_get_m(uint32_t m){
    uint32_t avail[26] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728};
    uint32_t i;
    for (i=0; i<26; i++) {
        if (avail[i] >= m) {
            return i;
        }
    }
    return i-1;
}
