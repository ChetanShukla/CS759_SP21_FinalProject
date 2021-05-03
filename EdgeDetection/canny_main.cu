#include "canny_cpu.cuh"
// #include "global.hpp"
#include "pixel.cuh"
#include "canny.cuh"
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

char type[10];
int intensity;

#define BLOCK_SIZE 32
#define WA 256   
#define HA 256     
#define HC 7    
#define WC 7
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)

// int hi;
// int lo;
// double sig;

void prepare_mask_arrays(float* maskx, float* masky, size_t dimension, int sigma) {

    int cent = dimension/2;

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

float* getPixelsFromPngImage(unsigned char *img, int width, int height, int channels) {

    size_t img_size = width * height * channels;
    float *pixels = (float*) malloc(sizeof(float) * img_size);

    unsigned int i = 0, j = 0;
    for(unsigned char *p = img; p != img + img_size; p += channels) {
        *(pixels + i) = (float) *p;
        i++;
    }

    return pixels;  
}

void getNormalisedMagnitudeMatrix(float* mag, unsigned int height, unsigned int width) {

    float maxVal = 0.0;
    unsigned int i, j;
	for (i = 0; i < height; i++) {
		for(j = 0; j < width; j++) {
            if (mag[i * width + j] > maxVal) {
                maxVal = mag[i * width + j];     
            }
		}
	}

	// Make sure all the magnitude values are between 0-255
    for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
            mag[i * width + j] = mag[i * width + j] / maxVal * 255;
            
    return;        
}

void printArrayForDebugging(float *arr, int height, int width) {
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            printf("%f ", arr[i*width + j]);
        }
        printf("\n");
    }
}

// =================================== Magnitude Image ==================================
// We'll start by filling in the mask values using the Gausian 1st derivative. Next, do a
// scanning convolution on the input pic matrix. This will give us the Δy and Δx matrices
// Finally, take the sqrt of the sum of Δy^2 and Δx^2 to find the magnitude.
// =================================== Magnitude Image ==================================
void magnitude_matrix(float *pic, float *mag, float *x, float *y, float *maskx, float *masky, double sig, int height, int width) {
    
    int dim = 6 * sig + 1, cent = dim / 2;

    printf("Pic: \n");
    printArrayForDebugging(pic, height, width);

	// Scanning convolution
	float sumx, sumy;
	for (int i = 0; i < height; i++)
	{ 
		for (int j = 0; j < width; j++)
		{
			sumx = 0;
			sumy = 0;

			// This is the convolution
			for (int p = -cent; p <= cent; p++)
			{
				for (int q = -cent; q <= cent; q++)
				{
					//if ((i+p) < 0 || (j+q) < 0 || (i+p) >= height || (j+q) >= width)
					//	continue;
                    
                    int rowIndex = i+p;
                    int colIndex = j+q;  
                    int maskRowIndex = p + cent;
                    int maskColIndex = q + cent; 
                    
                    if ((rowIndex * width + colIndex) < 0 || (maskRowIndex * width + maskColIndex) < 0 
                            || (rowIndex * width + colIndex) >= height*width 
                            || (maskRowIndex * width + maskColIndex) >= height*width)
						continue; 
                    
                    sumx += pic[rowIndex * width + colIndex] * maskx[maskRowIndex * dim + maskColIndex];
					sumy += pic[rowIndex * width + colIndex] * masky[maskRowIndex * dim + maskColIndex];
				}
			}
			
			// Store convolution result in respective matrix
			x[i * width + j] = sumx;
			y[i * width + j] = sumy;
		}
    }
    
    printf("\n\n X Matrix : \n");
    printArrayForDebugging(x, height, width);
    printf("\n\n Y Matrix : \n");
    printArrayForDebugging(y, height, width);    

	// Find magnitude and maxVal, then store it in the 'mag' matrix
	double mags;
	double maxVal = 0;
	for (int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			mags = sqrt((x[i * width + j] * x[i * width + j]) + (y[i * width + j] * y[i * width + j]));

			if (mags > maxVal)
				maxVal = mags;

			mag[i * width + j] = mags;
		}
    }
    
    printf("maxVal: %f\n", maxVal);

	// Make sure all the magnitude values are between 0-255
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
            if (maxVal != 0)    
                mag[i * width + j] = mag[i * width + j] / maxVal * 255;

	return;
}

int main(int argc, char **argv)
{
	// Exit program if proper arguments are not provided by user
    /*
    if (argc != 4)
	{
		cout << "Proper syntax: ./a.out <input_filename> <high_threshold> <sigma_value>" << endl;
		return 0;
    }
    */

    double sig = 1;
    int dim = 6 * sig + 1, cent = dim / 2;

    const int hi = 50;
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
        string path = "./processed_images/" + filename;

        cout << "\nProcessing Image: " << path << "\n\n";

        unsigned char *img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if(img == NULL) {
            printf("Error in loading the image\n");
            exit(1);
        }
        // printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

        // Step 2 : PNG image to 2D matrix
        float *pixels = getPixelsFromPngImage(img, width, height, channels);
        unsigned int i = 0, j = 0; 
        
        /* 
        printf("\n\nLet the magic begin!\n");

        for (i=0; i<height; i++) {
            for (j=0; j<width; j++) {
                printf("%u ", pixels[i*height + j]);
            }
            printf("\n");
        }
        */

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
        // convolution_kernel <<< grid, threads >>> (dev_pixels, dev_x, dev_mask_x, height, width, height, width, dim);
        cudaEventRecord(stop_G);

	    cudaEventRecord(start_G);
	    // convolution_kernel <<< grid, threads >>> (dev_pixels, dev_y, dev_mask_y, height, width, height, width, dim);
        cudaEventRecord(stop_G);

        // convolve(pixels, dev_x, dev_y, dev_mask_x, dev_mask_y, height, width, height, width, dim); 

        // After this step, we'll get the convolution in x direction and y direction in
        // the arrays x and y, which would later be used to generate vector of peaks
        // which in turn is used for creating the final array. 

        // Copy back the contents of dev_y to the host
        error = cudaMemcpy(y, dev_y, mask_memory_size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("Copying back y[] to the host failed");
            return EXIT_FAILURE;
        }

        // Copy back the contents of dev_x to the host
        error = cudaMemcpy(x, dev_x, mask_memory_size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("Copying back x[] to the host failed");
            return EXIT_FAILURE;
        }  

        float* mag = new float[height*width];

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

        // magnitude_matrix_kernel <<<num_blocks, threads_per_block>>> (dev_mag, dev_x, dev_y, height, width);

        // Copy back the contents of dev_mag to the host
        /*
        error = cudaMemcpy(mag, dev_mag, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            printf("Copying back mag[] to the host failed");
            return EXIT_FAILURE;
        }

        getNormalisedMagnitudeMatrix(mag, height, width);  
        */

        printf("\n\nMagnitude Matrix Before: \n");
        for (i=0; i<height; i++) {
            for (j=0; j<width; j++) {
                printf("%f ", mag[i*width + j]);
            }
            printf("\n");
        }
        magnitude_matrix(pixels, mag, x, y, maskx, masky, sig, height, width);

        printf("\n\nMagnitude Matrix After: \n");
        for (i=0; i<height; i++) {
            for (j=0; j<width; j++) {
                printf("%f ", mag[i*width + j]);
            }
            printf("\n");
        }


        // Step 5: Get all the peaks and store them in a vector
        unordered_map<Pixel*, bool> peaks;
        vector<Pixel*> vector_of_peaks = peak_detection(mag, peaks, x, y, height, width);


        // Step 6: Creation of the final image matrix using the magnitude matrix and
        // Recursive Double Thresholding
        uint8_t* final = new uint8_t[img_size];

        // Go through the vector and call the recursive function and each point. If the value
        // in the mag matrix is hi, then immediately accept it in final. If lo, then immediately
        // reject. If between lo and hi, then check if it's next to a hi pixel using recursion
        unordered_map<Pixel*, bool>  visited;
        int a, b;
        for (int i = 0; i < vector_of_peaks.size(); i++)
        {
            a = vector_of_peaks.at(i)->x;
            b = vector_of_peaks.at(i)->y;

            if (mag[a * width + b] >= hi)
                final[a * width + b] = 255;
            else if (mag[a * width + b] < lo)
                final[a * width + b] = 0;
            else
                recursiveDoubleThresholding(mag, final, visited, peaks, a, b, 0, width, height, lo);
        }

        /*
        printf("\n\nLet the magic begin!\n");

        for (i=0; i<height; i++) {
            for (j=0; j<width; j++) {
                printf("%u ", final[i*width + j]);
            }
            printf("\n");
        }
        */
        
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
        

        // Convert the input image to output image
        unsigned char *output_img = (unsigned char *)malloc(img_size);
        if(output_img == NULL) {
            printf("Unable to allocate memory for the output image.\n");
            exit(1);
        }

        i = 0;
        for(unsigned char *pg = output_img; i < img_size; i += channels, pg += channels) {
            *pg = final[i];     
        }

        string output_path = "./processed_images/output_" + filename;
        stbi_write_png(output_path.c_str(), width, height, channels, output_img, width * channels);
        // stbi_write_jpg("sky2.jpg", width, height, channels, img, 100);

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

	// Exit program if file doesn't open
	// string filename(argv[1]);
	// string path = "./input_images/" + filename;
	// ifstream infile(path, ios::binary);
	// if (!infile.is_open())
	// {
	// 	cout << "File " << path << " not found in directory." << endl;
	// 	return 0;
	// }	

	// Opening output files
	// ofstream img1("./output_images/canny_mag.pgm", ios::binary);
	// ofstream img2("./output_images/canny_peaks.pgm", ios::binary);		
	// ofstream img3("./output_images/canny_final.pgm", ios::binary);

	// ::hi = stoi(argv[2]);
	// ::lo = .35 * hi;
	// ::sig = stoi(argv[3]);

	// Storing header information and copying into the new ouput images
	// infile >> ::type >> width >> height >> ::intensity;
	// img1 << type << endl << width << " " << height << endl << intensity << endl;
	// img2 << type << endl << width << " " << height << endl << intensity << endl;
	// img3 << type << endl << width << " " << height << endl << intensity << endl;

	// These matrices will hold the integer values of the input image and masks.
	// I'm dynamically allocating arrays to easily pass them into functions.
	/*
    double **pic = new double*[height], **mag = new double*[height], **final = new double*[height];
	double **x = new double*[height], **y = new double*[height];

	
    for (int i = 0; i < height; i++)
	{
		pic[i] = new double[width];
		mag[i] = new double[width];
		final[i] = new double[width];
		x[i] = new double[width];
		y[i] = new double[width];
	}
    */

	// Reading in the input image as integers
	/*
    for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			pic[i][j] = (int)infile.get();

    convolve(pic, x, y, maskx, masky, height, width, height, width);        
    */
	// Create the magniute matrix
	// magnitude_matrix(pic, mag, x, y);


	

	// ================================= IMAGE OUTPUT =================================
	// Outputting the 'mag' matrix to img1. It's very important to cast it to a char.
	// To make sure that the decimal doesn't produce any wonky results, cast to an int
	// ================================= IMAGE OUTPUT =================================
	/*
    for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img1 << (char)((int)mag[i][j]);

	// Outputting the points stored in the vector to img2
	int k = 0;
	for (int i = 0; i < vector_of_peaks.size(); i++)
	{
		while(k++ != (vector_of_peaks.at(i)->x * height + vector_of_peaks.at(i)->y - 1))
			img2 << (char)(0);

		img2 << (char)(255);
	}

	// Output the 'final' matrix to img1
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img3 << (char)((int)final[i][j]);		
    */
	return 0;
}