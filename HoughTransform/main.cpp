#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>

#include "hough.cuh"

using namespace std;

int main()
{
	const int image_size = 256;
	uint8_t* image = new uint8_t[image_size * image_size];

	ifstream my_file("C:\\Users\\djkong7\\Documents\\GitHub\\CS759_SP21_FinalProject\\processed_images\\edges\\binary\\image-1", ios::in | ios::binary);
	if (my_file.is_open()) {
		my_file.read((char*)image, image_size * image_size);
		my_file.close();
	}
	else {
		cout << "File not opened";
		return 0;
	}

	/*for (int i = 0; i < image_size; i++) {
		for (int j = 0; j < image_size; j++) {
			printf("%d, ", image[i * image_size + j]);
		}
		cout << endl;
	}*/

	cudaError_t cuda_stat;
	uint8_t* dev_image;
	uint8_t* dev_edges;
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
	cuda_stat = cudaMalloc((void**)&dev_edges, sizeof(uint8_t) * image_size * image_size);
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

	find_edge_points << <num_blocks, threads_per_block, threads_per_block * sizeof(uint8_t) >> > (dev_image, image_size, dev_edges, dev_edges_len);
	hough << <1, 1 >> > (dev_edges, dev_edges_len);
	cudaDeviceSynchronize();

	cudaFree(dev_image);
	cudaFree(dev_edges);
	cudaFree(dev_edges_len);
	delete[] image;
	return 0;
}