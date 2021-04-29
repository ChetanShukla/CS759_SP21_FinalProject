#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "trig.cuh"
__global__ void accumulate_edge_points(uint8_t* image, int image_size, uint8_t* edges_x, uint8_t* edges_y, unsigned int* edges_len);
__global__ void hough(uint8_t* edges_x, uint8_t* edges_y, unsigned int* edges_len, float* global_sin, float* global_cos, int* global_acc);