#include <string>
#include <fstream>
#include "constants.cuh"
#include "canny.cuh"
#include "hough.cuh"

#define NUM_IMAGES 100

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int main() {
	int img_width, img_height, img_channels;
	const string input_dir = "../processed_images/segmented/";

#ifdef _WIN32
	const string output_dir = "..\\processed_images\\hough\\binary\\";
#else
	const string output_dir = "../processed_images/hough/binary/";
#endif

	for (int image_count = 1; image_count <= NUM_IMAGES; image_count++) {
		string filename = "image-" + to_string(image_count) + ".png";

		string full_path = input_dir + filename;

		if (DEBUG) {
			printf("Processing image: %s\n", full_path.c_str());
		}

		uint8_t* img = stbi_load(full_path.c_str(), &img_width, &img_height, &img_channels, 0);
		if (img == NULL || img_width != IMAGE_WIDTH || img_height != IMAGE_HEIGHT || img_channels != 1) {
			printf("Error in loading the image\n");
			exit(1);
		}

		uint8_t* edges = new uint8_t[IMAGE_WIDTH * IMAGE_HEIGHT]{ 0 };

		int canny_retval = canny_main(img, edges);
		if (canny_retval != 0) {
			exit(canny_retval);
		}

		string output_path = "../processed_images/edges/" + filename;
		stbi_write_png(output_path.c_str(), IMAGE_WIDTH, IMAGE_HEIGHT, 1, edges, IMAGE_WIDTH);

		int* accumulator = new int[TOTAL_ACC_SIZE];
		int hough_retval = hough_main(edges, accumulator);
		if (hough_retval != 0) {
			exit(hough_retval);
		}

		ofstream my_file_out((output_dir + "image-" + to_string(image_count) + "-out"), ios::out | ios::binary);
		if (my_file_out.is_open()) {
			my_file_out.write((char*)accumulator, sizeof(int) * TOTAL_ACC_SIZE);
			my_file_out.close();
		}
		else {
			printf("Output file not opened\n");
			return 1;
		}

		stbi_image_free(img);
		delete[] edges;
		delete[] accumulator;
	}
	return 0;
}