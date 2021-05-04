#include "canny_cpu.cuh"
#include "pixel.cuh"
#include "canny.cuh"
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#define BLOCK_SIZE 32
#define WA 256
#define HA 256
#define HC 7
#define WC 7
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)

void printArrayForDebugging(float* arr, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f ", arr[i * width + j]);
		}
		printf("\n");
	}
}

void prepare_mask_arrays(float* maskx, float* masky, size_t dimension, int sigma) {
	int cent = dimension / 2;

	// Use the Gausian 1st derivative formula to fill in the mask values
	float denominator = 2 * sigma * sigma;

	for (int p = -cent; p <= cent; p++) {
		for (int q = -cent; q <= cent; q++) {
			float numerator = (p * p + q * q);

			int rowIndex = p + cent;
			int colIndex = q + cent;

			// maskx[p+cent][q+cent] = q * exp(-1 * (numerator / denominator))
			maskx[rowIndex * dimension + colIndex] = q * exp(-1 * (numerator / denominator));

			// masky[p+cent][q+cent] = p * exp(-1 * (numerator / denominator))
			masky[rowIndex * dimension + colIndex] = p * exp(-1 * (numerator / denominator));
		}
	}
}

float* getPixelsFromPngImage(unsigned char* img, int width, int height, int channels) {
	size_t img_size = width * height * channels;
	float* pixels = (float*)malloc(sizeof(float) * img_size);

	unsigned int i = 0;
	for (unsigned char* p = img; p != img + img_size; p += channels) {
		*(pixels + i) = (float)*p;
		i++;
	}

	return pixels;
}

void getNormalisedMagnitudeMatrix(float* mag, unsigned int height, unsigned int width) {
	// printf("\n\nMagnitude matrix before normalisation:\n");
	// printArrayForDebugging(mag, height, width);

	float maxVal = 0.0;
	unsigned int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (mag[i * width + j] > maxVal) {
				maxVal = mag[i * width + j];
			}
		}
	}

	// printf("maxVal: %f\n", maxVal);

	// Make sure all the magnitude values are between 0-255
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			mag[i * width + j] = mag[i * width + j] / maxVal * 255;

	return;
}

int main(int argc, char** argv) {
	double sig = 1;
	int dim = 6 * sig + 1, cent = dim / 2;

	const int hi = 40;
	const int lo = .35 * hi;

	cudaError_t error;
	unsigned int mask_size = dim * dim;
	unsigned int mask_memory_size = sizeof(float) * mask_size;

	// float maskx[dim][dim], masky[dim][dim]
	float* maskx = (float*)malloc(mask_memory_size);
	float* masky = (float*)malloc(mask_memory_size);

	// Step 1: Creation of mask arrays using the Gaussian derivative formula
	prepare_mask_arrays(maskx, masky, dim, sig);

	float* dev_mask_x;
	float* dev_mask_y;

	// allocate memory for x[] on the device(GPU)
	error = cudaMalloc((void**)&dev_mask_x, mask_memory_size);
	if (error != cudaSuccess) {
		printf("Allocation for mask_x[] on the device memory failed!");
		return EXIT_FAILURE;
	}

	// allocate memory for y[] on the device(GPU)
	error = cudaMalloc((void**)&dev_mask_y, mask_memory_size);
	if (error != cudaSuccess) {
		printf("Allocation for mask_y[] on the device memory failed!");
		return EXIT_FAILURE;
	}

	// Put the mask_x on the device
	error = cudaMemcpy(dev_mask_x, maskx, mask_memory_size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Moving the mask_x[] to device failed");
		return EXIT_FAILURE;
	}

	// Put the mask_y on the device
	error = cudaMemcpy(dev_mask_y, masky, mask_memory_size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Moving the mask_y[] to device failed");
		return EXIT_FAILURE;
	}

	int width, height, channels, total_images = 2;

	for (int image_count = 1; image_count <= total_images; image_count++) {
		/*
			Reading the required PNG image from the images folder which would
			be processed in the current iteration.
		*/

		string filename = "image-" + to_string(image_count) + ".png";
		string path = "../processed_images/gray/" + filename;

		//cout << "\nProcessing Image: " << path << "\n\n";

		unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 0);
		if (img == NULL) {
			printf("Error in loading the image\n");
			exit(1);
		}
		// printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

		// Step 2 : PNG image to 2D matrix
		float* pixels = getPixelsFromPngImage(img, width, height, channels);

		cudaEvent_t start_G, stop_G;

		cudaEventCreate(&start_G);
		cudaEventCreate(&stop_G);

		float* x = new float[width * height];
		float* y = new float[width * height];

		float* dev_x;
		float* dev_y;
		float* dev_pixels;

		unsigned int mem_size_x = sizeof(float) * width * height;
		unsigned int mem_size_y = sizeof(float) * width * height;
		unsigned int img_size = width * height * channels;
		unsigned int mem_size_img = sizeof(float) * img_size;

		// allocate memory for x[] on the device(GPU)
		error = cudaMalloc((void**)&dev_x, mem_size_x);
		if (error != cudaSuccess) {
			printf("Allocation for x[] on the device memory failed!");
			return EXIT_FAILURE;
		}

		// allocate memory for y[] on the device(GPU)
		error = cudaMalloc((void**)&dev_y, mem_size_y);
		if (error != cudaSuccess) {
			printf("Allocation for y[] on the device memory failed!");
			return EXIT_FAILURE;
		}

		// allocate memory for pixels[] on the device(GPU)
		error = cudaMalloc((void**)&dev_pixels, mem_size_img);
		if (error != cudaSuccess) {
			printf("Allocation for pixels[] on the device memory failed!");
			return EXIT_FAILURE;
		}

		// Copying the content of pixels[] to the device(GPU)
		error = cudaMemcpy(dev_pixels, pixels, mem_size_img, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
			printf("Copying the pixels[] to device failed");
			return EXIT_FAILURE;
		}

		// Step 3: Convolution of the image matrix with the mask arrays in the two dimensions

		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));

		cudaEventRecord(start_G);
		convolution_kernel << < grid, threads >> > (dev_pixels, dev_x, dev_mask_x, height, width, height, width, dim);
		cudaEventRecord(stop_G);

		cudaEventRecord(start_G);
		convolution_kernel << < grid, threads >> > (dev_pixels, dev_y, dev_mask_y, height, width, height, width, dim);
		cudaEventRecord(stop_G);

		float* mag = new float[height * width];

		// After this step, we'll get the convolution in x direction and y direction in
		// the arrays x and y, which would later be used to generate vector of peaks
		// which in turn is used for creating the final array.

		// Copy back the contents of dev_y to the host
		error = cudaMemcpy(y, dev_y, mem_size_y, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("Copying back y[] to the host failed");
			return EXIT_FAILURE;
		}

		// Copy back the contents of dev_x to the host
		error = cudaMemcpy(x, dev_x, mem_size_x, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("Copying back x[] to the host failed");
			return EXIT_FAILURE;
		}

		float* dev_mag;

		// allocate image memory on the device(GPU)
		error = cudaMalloc((void**)&dev_mag, sizeof(float) * height * width);
		if (error != cudaSuccess) {
			printf("device image memory allocation failed");
			return EXIT_FAILURE;
		}

		const int threads_per_block = 256;
		int num_blocks = (img_size + threads_per_block - 1) / threads_per_block;

		// Step 4: Get the magnitude matrix using the x[] and y[] that we got from the previous step

		magnitude_matrix_kernel << <num_blocks, threads_per_block >> > (dev_mag, dev_x, dev_y, height, width);

		// Copy back the contents of dev_mag to the host

		error = cudaMemcpy(mag, dev_mag, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {
			printf("Copying back mag[] to the host failed");
			return EXIT_FAILURE;
		}

		getNormalisedMagnitudeMatrix(mag, height, width);

		//printf("\n\nMagnitude Matrix After: \n");
		//printArrayForDebugging(mag, height, width);

		// Step 5: Get all the peaks and store them in a vector
		unordered_map<Pixel*, bool> peaks;
		vector<Pixel*> vector_of_peaks = peak_detection(mag, peaks, x, y, height, width);

		// Step 6: Creation of the final image matrix using the magnitude matrix and
		// Recursive Double Thresholding
		uint8_t* final = new uint8_t[img_size]{ 0 };

		// Go through the vector and call the recursive function and each point. If the value
		// in the mag matrix is hi, then immediately accept it in final. If lo, then immediately
		// reject. If between lo and hi, then check if it's next to a hi pixel using recursion
		unordered_map<Pixel*, bool>  visited;
		int a, b;
		for (int i = 0; i < vector_of_peaks.size(); i++)
		{
			a = vector_of_peaks.at(i)->x;
			b = vector_of_peaks.at(i)->y;

			if (mag[a * width + b] >= hi) {
				final[a * width + b] = 255;
			}
			else if (mag[a * width + b] < lo) {
				final[a * width + b] = 0;
			}
			else {
				recursiveDoubleThresholding(mag, final, visited, peaks, a, b, 0, width, height, lo);
			}
		}

		/*printf("\n\nFinal Image Matrix: \n");
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%u ", final[i * width + j]);
			}
			printf("\n");
		}*/

		// Final step : Storing the final image matrix in the Device/GPU global memory
		// for further processing in the Hough transform step

		uint8_t* dev_final_image;

		// allocate image memory on the device(GPU)
		error = cudaMalloc((void**)&dev_final_image, sizeof(uint8_t) * img_size);
		if (error != cudaSuccess) {
			printf("device image memory allocation failed");
			return EXIT_FAILURE;
		}

		// put the image on the device
		error = cudaMemcpy(dev_final_image, final, sizeof(uint8_t) * img_size, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
			printf("image move to device failed");
			return EXIT_FAILURE;
		}

		//// Convert the input image to output image
		//unsigned char* output_img = (unsigned char*)malloc(img_size);
		//if (output_img == NULL) {
		//	printf("Unable to allocate memory for the output image.\n");
		//	exit(1);
		//}

		//unsigned int i = 0;
		//for (unsigned char* pg = output_img; i < img_size; i += channels, pg += channels) {
		//	*pg = final[i];
		//}

		string output_path = "../processed_images/edges/" + filename;
		stbi_write_png(output_path.c_str(), width, height, channels, final, width * channels);

		stbi_image_free(img);

		delete[] x;
		delete[] y;
		delete[] final;
		delete[] mag;
		cudaFree(dev_x);
		cudaFree(dev_y);
		cudaFree(dev_pixels);
		cudaFree(dev_mag);
		cudaFree(dev_final_image);
		free(pixels);
	}

	free(maskx);
	free(masky);
	cudaFree(dev_mask_x);
	cudaFree(dev_mask_y);

	return 0;
}