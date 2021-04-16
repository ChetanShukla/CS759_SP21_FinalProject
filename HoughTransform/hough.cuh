#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
__global__ void find_edge_points(uint8_t* image, int image_size, uint8_t* edges, unsigned int* edges_len);
__global__ void hough(uint8_t* edges, unsigned int* edges_len);