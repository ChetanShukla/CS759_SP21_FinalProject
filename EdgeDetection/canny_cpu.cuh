#ifndef CANNY_CPU_CUH
#define CANNY_CPU_CUH

#include "pixel.cuh"
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std;

std::vector<Pixel*> peak_detection(float *mag, unordered_map<Pixel*, bool> peaks, float *x, float *y, 
                                    const int height, const int width);

void recursiveDoubleThresholding(float *mag, uint8_t *final, unordered_map<Pixel*, bool> visited, unordered_map<Pixel*, bool> peaks,
                                    int i, int j, int flag, const int width, const int height, const int lo);

#endif