#include "pixel.cuh"

using namespace std;

// Pixel constructor
Pixel::Pixel(int a, int b) {
	x = a;
	y = b;
}

Pixel::Pixel() {
	x = 0;
	y = 0;
}
