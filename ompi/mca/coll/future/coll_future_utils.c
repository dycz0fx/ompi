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
