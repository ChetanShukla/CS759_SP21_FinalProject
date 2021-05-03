#include "canny.hpp"
#include "global.hpp"
#include "pixel.hpp"
#include "canny.cuh"
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

char type[10];
int intensity;
int hi;
int lo;
double sig;

int main(int argc, char **argv)
{
	// Exit program if proper arguments are not provided by user
	if (argc != 4)
	{
		cout << "Proper syntax: ./a.out <input_filename> <high_threshold> <sigma_value>" << endl;
		return 0;
	}

    int width, height, channels, total_images = 100;

    for (int image_count = 1; image_count <= total_images; image_count++) {
        string filename = "image-" + to_string(image_count) + ".png";
        string path = "./processed_images/" + filename;

        cout << "\nProcessing Image: " << path << "\n\n";

        unsigned char *img = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if(img == NULL) {
            printf("Error in loading the image\n");
            exit(1);
        }
        printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

        size_t gray_img_size = width * height * channels;
        size_t img_size = width * height * channels;
        int gray_channels = channels;
        unsigned char *gray_img = (unsigned char*) malloc(gray_img_size);
        unsigned int *pixels = (unsigned int*) malloc(gray_img_size);

        unsigned int i = 0, j = 0;
        for(unsigned char *p = img, *pg = gray_img; p != img + img_size; p += channels, pg += gray_channels) {
            // *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
            *pg = (uint8_t) *p;
            printf("%u ", *pg);

            *(pixels + i) = (uint8_t) *pg;
            i++;
            
            /* if(channels == 4) {
                *(pg + 1) = *(p + 3);
            } */
        }
        printf("\n\nLet the magic begin!\n");

        for (i=0; i<height; i++) {
            for (j=0; j<width; j++) {
                printf("%u ", pixels[i*height + j]);
            }
            printf("\n");
        }

        // stbi_write_png("sky.png", width, height, channels, img, width * channels);
        // stbi_write_jpg("sky2.jpg", width, height, channels, img, 100);

        stbi_image_free(img);

    }

	// Exit program if file doesn't open
	string filename(argv[1]);
	string path = "./input_images/" + filename;
	ifstream infile(path, ios::binary);
	if (!infile.is_open())
	{
		cout << "File " << path << " not found in directory." << endl;
		return 0;
	}	

	// Opening output files
	ofstream img1("./output_images/canny_mag.pgm", ios::binary);
	ofstream img2("./output_images/canny_peaks.pgm", ios::binary);		
	ofstream img3("./output_images/canny_final.pgm", ios::binary);

	::hi = stoi(argv[2]);
	::lo = .35 * hi;
	::sig = stoi(argv[3]);

	// Storing header information and copying into the new ouput images
	infile >> ::type >> width >> height >> ::intensity;
	img1 << type << endl << width << " " << height << endl << intensity << endl;
	img2 << type << endl << width << " " << height << endl << intensity << endl;
	img3 << type << endl << width << " " << height << endl << intensity << endl;

	// These matrices will hold the integer values of the input image and masks.
	// I'm dynamically allocating arrays to easily pass them into functions.
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

	// Reading in the input image as integers
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			pic[i][j] = (int)infile.get();

    convolve(pic, x, y, height, width, height, width);        

	// Create the magniute matrix
	// magnitude_matrix(pic, mag, x, y);

	// Get all the peaks and store them in vector
	unordered_map<Pixel*, bool> peaks;
	vector<Pixel*> v = peak_detection(mag, peaks, x, y);

	// Go through the vector and call the recursive function and each point. If the value
	// in the mag matrix is hi, then immediately accept it in final. If lo, then immediately
	// reject. If between lo and hi, then check if it's next to a hi pixel using recursion
	unordered_map<Pixel*, bool>  visited;
	int a, b;
	for (int i = 0; i < v.size(); i++)
	{
		a = v.at(i)->x;
		b = v.at(i)->y;

		if (mag[a][b] >= hi)
			final[a][b] = 255;
		else if (mag[a][b] < lo)
			final[a][b] = 0;
		else
			recursiveDoubleThresholding(mag, final, visited, peaks, a, b, 0);
	}

	// ================================= IMAGE OUTPUT =================================
	// Outputting the 'mag' matrix to img1. It's very important to cast it to a char.
	// To make sure that the decimal doesn't produce any wonky results, cast to an int
	// ================================= IMAGE OUTPUT =================================
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img1 << (char)((int)mag[i][j]);

	// Outputting the points stored in the vector to img2
	int k = 0;
	for (int i = 0; i < v.size(); i++)
	{
		while(k++ != (v.at(i)->x * height + v.at(i)->y - 1))
			img2 << (char)(0);

		img2 << (char)(255);
	}

	// Output the 'final' matrix to img1
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img3 << (char)((int)final[i][j]);		

	return 0;
}