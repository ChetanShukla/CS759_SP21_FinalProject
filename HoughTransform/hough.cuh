#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
__global__ void accumulate_edge_points(uint8_t* image, int image_size, int* edges_x, int* edges_y, int* edges_len);
__global__ void hough(int* edges_x, int* edges_y, int* edges_len, int* global_acc);