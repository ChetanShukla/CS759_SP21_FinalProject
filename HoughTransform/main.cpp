#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

#include "hough.cuh"

using namespace std;

int main()
{
	const int image_size = 256;
	uint8_t* image = new uint8_t[image_size * image_size];
	float accumulate_time = 0.0;
	float hough_time = 0.0;
	int num_images = 100;

	for (int z = 1; z <= num_images; z++) {
		ifstream my_file("C:\\Users\\djkong7\\Documents\\GitHub\\CS759_SP21_FinalProject\\processed_images\\edges\\binary\\image-" + to_string(z), ios::in | ios::binary);
		if (my_file.is_open()) {
			my_file.read((char*)image, image_size * image_size);
			my_file.close();
		}
		else {
			cout << "File not opened";
			return 0;
		}

		cudaError_t cuda_stat;
		uint8_t* dev_image;
		uint8_t* dev_edges_x;
		uint8_t* dev_edges_y;
		unsigned int* dev_edges_len;

		// allocate image memory on the device(GPU)
		cuda_stat = cudaMalloc((void**)&dev_image, sizeof(uint8_t) * image_size * image_size);
		if (cuda_stat != cudaSuccess) {
			printf("device image memory allocation failed");
			return EXIT_FAILURE;
		}

		// put the image on the device
		cuda_stat = cudaMemcpy(dev_image, image, sizeof(uint8_t) * image_size * image_size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			printf("image move to device failed");
			return EXIT_FAILURE;
		}

		// allocate edges memory on the device(GPU)
		cuda_stat = cudaMalloc((void**)&dev_edges_x, sizeof(uint8_t) * image_size * image_size);
		if (cuda_stat != cudaSuccess) {
			printf("device edges memory allocation failed");
			return EXIT_FAILURE;
		}

		// allocate edges memory on the device(GPU)
		cuda_stat = cudaMalloc((void**)&dev_edges_y, sizeof(uint8_t) * image_size * image_size);
		if (cuda_stat != cudaSuccess) {
			printf("device edges memory allocation failed");
			return EXIT_FAILURE;
		}

		// allocate edges length memory on the device(GPU)
		cuda_stat = cudaMalloc((void**)&dev_edges_len, sizeof(int));
		if (cuda_stat != cudaSuccess) {
			printf("device edges length memory allocation failed");
			return EXIT_FAILURE;
		}
		// Initialize the global points length to 0
		cuda_stat = cudaMemset((void*)dev_edges_len, 0, sizeof(int));
		if (cuda_stat != cudaSuccess) {
			printf("device edges length memset failed");
			return EXIT_FAILURE;
		}

		const int threads_per_block = 256;
		int num_blocks = (image_size * image_size + threads_per_block - 1) / threads_per_block;

		// Start timer code
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		accumulate_edge_points << <num_blocks, threads_per_block >> > (dev_image, image_size, dev_edges_x, dev_edges_y, dev_edges_len);

		// End timer code
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		// Get the elapsed time in milliseconds
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		//printf("Edge array creation: %.3fms\n", ms);
		accumulate_time += ms;

		int* acc = new int[64 * 64 * 3];
		int* dev_acc;
		// allocate accumulator memory on the device(GPU)
		cuda_stat = cudaMalloc((void**)&dev_acc, sizeof(int) * 64 * 64 * 3);
		if (cuda_stat != cudaSuccess) {
			printf("device accumulator memory allocation failed");
			return EXIT_FAILURE;
		}

		// Start timer code
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		hough << <3, 1024 >> > (dev_edges_x, dev_edges_y, dev_edges_len, dev_acc);

		// End timer code
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		// Get the elapsed time in milliseconds
		cudaEventElapsedTime(&ms, start, stop);
		//printf("Hough accumulation: %.3fms\n", ms);
		hough_time += ms;

		cudaDeviceSynchronize();
		// Get the accumulator from global memory
		cuda_stat = cudaMemcpy(acc, dev_acc, sizeof(int) * 64 * 64 * 3, cudaMemcpyDeviceToHost);
		if (cuda_stat != cudaSuccess) {
			printf("Accumulator move to host failed");
			return EXIT_FAILURE;
		}

		ofstream my_file_out(("C:\\Users\\djkong7\\Documents\\GitHub\\CS759_SP21_FinalProject\\processed_images\\hough\\binary\\image-" + to_string(z) + "-out"), ios::out | ios::binary);
		if (my_file_out.is_open()) {
			my_file_out.write((char*)acc, sizeof(int) * 64 * 64 * 3);
			my_file_out.close();
		}
		else {
			cout << "Output file not opened\n";
			return 0;
		}

		cudaFree(dev_image);
		cudaFree(dev_edges_x);
		cudaFree(dev_edges_y);
		cudaFree(dev_edges_len);
		cudaFree(dev_acc);
		delete[] acc;
	}
	delete[] image;

	printf("Average edge array creation: %.3fms\n", accumulate_time / num_images);
	printf("Average hough accumulation: %.3fms\n", hough_time / num_images);

	return 0;
}