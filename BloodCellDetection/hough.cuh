#ifndef HOUGH_CUH
#define HOUGH_CUH

#include "device_launch_parameters.h"
#include <stdint.h>

#define ACCUMULATOR_SIZE 64
#define NUM_RADIUS 3
#define TOTAL_ACC_SIZE ACCUMULATOR_SIZE * ACCUMULATOR_SIZE * NUM_RADIUS

__global__ void accumulate_edge_points(uint8_t* image, int image_size, int* edges_x, int* edges_y, int* edges_len);
__global__ void hough(int* edges_x, int* edges_y, int* edges_len, int* global_acc);
int hough_main(uint8_t* edges, int* accumulator);

#endif HOUGH_CUH