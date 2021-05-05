/*
Credit to https://github.com/sorazy/canny/ for these to functions.
Slight modifications were made for our use case.
*/
#include "canny_cpu.cuh"

using namespace std;

/***
 * ===============================> Peaks Detection <================================
 * Slope of given line = Δy/Δx. We have Δy and Δx from the scanning convolution
 * step before. We'll store all the points that are greater than both its neighbors
 * neighbors in the direction of the slope into a vector. We can calculate the
 * direction of the slope using the tan(x) function. We'll also store the peaks in a
 * HashMap for O(1) look-up later.
 * ================================> Peaks Detection <===============================
***/

vector<Pixel*> peak_detection(float* mag, unordered_map<Pixel*, bool> peaks, float* x, float* y, int height, int width) {
	double slope = 0;
	vector<Pixel*> points;
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			// The value of this can be used for [i][j]
			int equivalent1DCurrentIndex = i * width + j;

			// To avoid dividing by zero
			if (x[equivalent1DCurrentIndex] == 0)
				x[equivalent1DCurrentIndex] = 0.00001;

			slope = y[equivalent1DCurrentIndex] / x[equivalent1DCurrentIndex];
			Pixel* givenPoint = new Pixel(i, j);

			bool shouldInsert = false;

			// We're only looking for the peaks. If we're at a peak, store 255 in 'peaks'
			if (slope <= tan(22.5) && slope > tan(-22.5)) {
				if (mag[equivalent1DCurrentIndex] > mag[equivalent1DCurrentIndex - 1]
					&& mag[equivalent1DCurrentIndex] > mag[equivalent1DCurrentIndex + 1]) {
					shouldInsert = true;
				}
			}
			else if (slope <= tan(67.5) && slope > tan(22.5)) {
				if (mag[equivalent1DCurrentIndex] > mag[((i - 1) * width) + j - 1]
					&& mag[equivalent1DCurrentIndex] > mag[((i + 1) * width) + j + 1]) {
					shouldInsert = true;
				}
			}
			else if (slope <= tan(-22.5) && slope > tan(-67.5)) {
				if (mag[equivalent1DCurrentIndex] > mag[((i + 1) * width) + j - 1]
					&& mag[equivalent1DCurrentIndex] > mag[((i - 1) * width) + j + 1]) {
					shouldInsert = true;
				}
			}
			else {
				if (mag[equivalent1DCurrentIndex] > mag[((i - 1) * width) + j]
					&& mag[equivalent1DCurrentIndex] > mag[((i + 1) * width) + j]) {
					shouldInsert = true;
				}
			}

			if (shouldInsert == true) {
				points.push_back(givenPoint);
				peaks.insert(make_pair(givenPoint, true));
			}
		}
	}

	return points;
}

/***
 * ========================> Hysteresis & Double Thresholding <========================
 * The points passed into this function are coming from the peaks vector. We'll start
 * by searching around the current pixel for a pixel that made it to "final". If found,
 * then we'll recursively search for a "series" of pixels that are in the mid range and
 * switch all those to ON in final. We'll stop as soon as all the pixels are either
 * already processed or less than the 'lo' threshold.
 * ========================> Hysteresis & Double Thresholding <========================
***/
void recursiveDoubleThresholding(float* mag, uint8_t* final, unordered_map<Pixel*, bool> visited,
	unordered_map<Pixel*, bool> peaks, int a, int b, int flag,
	const int width, const int height, const int lo) {
	// If the pixel value is < lo, out-of-bounds, or at a point we've visited before,
	// then exit the funciton.
	if (mag[a * width + b] < lo || a < 0 || b < 0 || a >= height || b >= width)
		return;

	Pixel* givenPoint = new Pixel(a, b);
	if (visited.find(givenPoint) != visited.end())
		return;

	// Insert the current pixel so we know we've been here before.
	visited.insert(make_pair(givenPoint, true));

	// If flag = 0, that means that this is the first pixel of the "series" that
	// we're looking at. We're going to look for a pixel in "final" that's set to
	// ON. If we found one, assert the flag and break out of the loops.
	if (!flag) {
		for (int p = -1; p <= 1; p++) {
			for (int q = -1; q <= 1; q++) {
				int rowIndex = a + p;
				int colIndex = b + q;
				if (final[rowIndex * width + colIndex] == 255) {
					final[a * width + b] = 255;
					flag = 1;
					break;
				}
			}

			if (flag)
				break;
		}
	}

	// If flag is asserted, that means we found a pixel that's in final, all what
	// we have to do now is just search for pixels that are in the mid range.
	// Also, make sure that it's in the peaks to begin with.
	if (flag) {
		for (int p = -1; p <= 1; p++) {
			for (int q = -1; q <= 1; q++) {
				int rowIndex = a + p;
				int colIndex = b + q;

				Pixel* currentPoint = new Pixel(rowIndex, colIndex);

				if (mag[rowIndex * width + colIndex] >= lo && peaks.find(currentPoint) != peaks.end()) {
					recursiveDoubleThresholding(mag, final, visited, peaks, a + p, b + q, 1, width, height, lo);
					final[a * width + b] = 255;
				}
			}
		}
	}

	return;
}