#include <string>
#include "constants.cuh"
#include "canny_main.cuh"

#define NUM_IMAGES 1

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int main() {
	int img_width, img_height, img_channels;

	for (int image_count = 1; image_count <= NUM_IMAGES; image_count++) {
		string filename = "image-" + to_string(image_count) + ".png";
		string input_dir = "../processed_images/segmented/";
		string full_path = input_dir + filename;

		if (DEBUG) {
			printf("Processing image: %s\n", full_path.c_str());
		}

		uint8_t* img = stbi_load(full_path.c_str(), &img_width, &img_height, &img_channels, 0);
		if (img == NULL || img_width != IMAGE_WIDTH || img_height != IMAGE_HEIGHT || img_channels != 1) {
			printf("Error in loading the image\n");
			exit(1);
		}

		uint8_t* output = new uint8_t[IMAGE_WIDTH * IMAGE_HEIGHT]{ 0 };

		int idk = canny_main(img, output);
		stbi_image_free(img);

		string output_path = "../processed_images/edges/" + filename;
		stbi_write_png(output_path.c_str(), IMAGE_WIDTH, IMAGE_HEIGHT, 1, output, IMAGE_WIDTH);
	}
}