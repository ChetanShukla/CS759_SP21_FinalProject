#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "hough.cuh"
using namespace std;

#define IMAGE_SIZE 256
#define NUM_IMAGES 100
#define ACCUMULATOR_SIZE 64
#define NUM_RADIUS 3
#define TOTAL_ACC_SIZE ACCUMULATOR_SIZE * ACCUMULATOR_SIZE * NUM_RADIUS

int start()
{
	const string input_dir = "C:\\Users\\djkong7\\Documents\\GitHub\\CS759_SP21_FinalProject\\processed_images\\edges\\binary\\";
	const string output_dir = "C:\\Users\\djkong7\\Documents\\GitHub\\CS759_SP21_FinalProject\\processed_images\\hough\\binary\\";
	uint8_t* image = new uint8_t[IMAGE_SIZE * IMAGE_SIZE];
	float accumulate_time = 0.0;
	float hough_time = 0.0;

	cudaError_t cuda_stat;
	uint8_t* dev_image;
	int* dev_edges_x;
	int* dev_edges_y;
	int* dev_edges_len;
	int* acc = new int[TOTAL_ACC_SIZE];
	int* dev_acc;

	// allocate image memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_image, sizeof(uint8_t) * IMAGE_SIZE * IMAGE_SIZE);
	if (cuda_stat != cudaSuccess) {
		printf("device image memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate edges x memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges_x, sizeof(int) * IMAGE_SIZE * IMAGE_SIZE);
	if (cuda_stat != cudaSuccess) {
		printf("device edges x memory allocation failed");
		return EXIT_FAILURE;
	}

	// allocate edges y memory on the device(GPU)
	cuda_stat = cudaMalloc((void**)&dev_edges_y, sizeof(int) * IMAGE_SIZE * IMAGE_SIZE);
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

	for (int z = 1; z <= NUM_IMAGES; z++) {
		ifstream my_file(input_dir + "image-" + to_string(z), ios::in | ios::binary);
		if (my_file.is_open()) {
			my_file.read((char*)image, IMAGE_SIZE * IMAGE_SIZE);
			my_file.close();
		}
		else {
			cout << "File not opened";
			return 0;
		}

		// put the image on the device
		cuda_stat = cudaMemcpy(dev_image, image, sizeof(uint8_t) * IMAGE_SIZE * IMAGE_SIZE, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			printf("image move to device failed");
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

		accumulate_edge_points << <IMAGE_SIZE, IMAGE_SIZE >> > (dev_image, IMAGE_SIZE, dev_edges_x, dev_edges_y, dev_edges_len);

		// End timer code
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		// Get the elapsed time in milliseconds
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		//printf("Edge array creation: %.3fms\n", ms);
		accumulate_time += ms;

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
		//printf("Hough accumulation: %.3fms\n", ms);
		hough_time += ms;

		//Here just for testing.
		cudaDeviceSynchronize();

		// Get the accumulator from global memory
		cuda_stat = cudaMemcpy(acc, dev_acc, sizeof(int) * TOTAL_ACC_SIZE, cudaMemcpyDeviceToHost);
		if (cuda_stat != cudaSuccess) {
			printf("Accumulator move to host failed");
			return EXIT_FAILURE;
		}

		ofstream my_file_out((output_dir + "image-" + to_string(z) + "-out"), ios::out | ios::binary);
		if (my_file_out.is_open()) {
			my_file_out.write((char*)acc, sizeof(int) * TOTAL_ACC_SIZE);
			my_file_out.close();
		}
		else {
			cout << "Output file not opened\n";
			return 0;
		}
	}
	cudaFree(dev_image);
	cudaFree(dev_edges_x);
	cudaFree(dev_edges_y);
	cudaFree(dev_edges_len);
	cudaFree(dev_acc);
	delete[] acc;
	delete[] image;

	printf("Average edge array creation: %.3fms\n", accumulate_time / NUM_IMAGES);
	printf("Average hough accumulation: %.3fms\n", hough_time / NUM_IMAGES);

	return 0;
}