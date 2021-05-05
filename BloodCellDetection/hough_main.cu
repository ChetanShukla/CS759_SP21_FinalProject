#include <stdio.h>

#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "hough.cuh"
#include "constants.cuh"
using namespace std;

int hough_main(uint8_t* edges, int* accumulator)
{
	cudaError_t cuda_stat;
	uint8_t* dev_edges;
	int* dev_edges_x;
	int* dev_edges_y;
	int* dev_edges_len;
	int* dev_acc;

	// allocate edges memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges, sizeof(uint8_t) * IMAGE_WIDTH * IMAGE_HEIGHT);
	if (cuda_stat != cudaSuccess) {
		printf("device edges memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate edges x memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges_x, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT);
	if (cuda_stat != cudaSuccess) {
		printf("device edges x memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate edges y memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges_y, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT);
	if (cuda_stat != cudaSuccess) {
		printf("device edges y memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate edges length memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges_len, sizeof(int));
	if (cuda_stat != cudaSuccess) {
		printf("device edges length memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate accumulator memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_acc, sizeof(int) * TOTAL_ACC_SIZE);
	if (cuda_stat != cudaSuccess) {
		printf("device accumulator memory allocation failed");
		return EXIT_FAILURE;
	}

	// put the edges on the device
	cuda_stat = cudaMemcpy(dev_edges, edges, sizeof(uint8_t) * IMAGE_WIDTH * IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	if (cuda_stat != cudaSuccess) {
		printf("edges move to device failed");
		return EXIT_FAILURE;
	}

	// Initialize the global points length to 0
	cuda_stat = cudaMemset((void*)dev_edges_len, 0, sizeof(int));
	if (cuda_stat != cudaSuccess) {
		printf("device edges length memset failed");
		return EXIT_FAILURE;
	}

	// Start timer code
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	accumulate_edge_points << <IMAGE_WIDTH, IMAGE_HEIGHT >> > (dev_edges, IMAGE_WIDTH, dev_edges_x, dev_edges_y, dev_edges_len);

	// End timer code
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// Get the elapsed time in milliseconds
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	if (DEBUG) { printf("Edge array creation: %.3fms\n", ms); }

	// Start timer code
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	hough << <NUM_RADIUS, 1024 >> > (dev_edges_x, dev_edges_y, dev_edges_len, dev_acc);

	// End timer code
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// Get the elapsed time in milliseconds
	cudaEventElapsedTime(&ms, start, stop);
	if (DEBUG) { printf("Hough accumulation: %.3fms\n", ms); }

	// Get the accumulator from global memory
	cuda_stat = cudaMemcpy(accumulator, dev_acc, sizeof(int) * TOTAL_ACC_SIZE, cudaMemcpyDeviceToHost);
	if (cuda_stat != cudaSuccess) {
		printf("Accumulator move to host failed");
		return EXIT_FAILURE;
	}

	cudaFree(dev_edges);
	cudaFree(dev_edges_x);
	cudaFree(dev_edges_y);
	cudaFree(dev_edges_len);
	cudaFree(dev_acc);

	return 0;
}