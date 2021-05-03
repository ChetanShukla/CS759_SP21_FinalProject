#ifndef CANNY_HPP
#define CANNY_HPP

#include "pixel.hpp"
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std;

std::vector<Pixel*> peak_detection(double *mag, unordered_map<Pixel*, bool> peaks, double *x, double *y, 
                                    const int height, const int width);

void recursiveDoubleThresholding(double *mag, double *final, unordered_map<Pixel*, bool> visited, 
                                    unordered_map<Pixel*, bool> peaks, int i, int j, int flag, const int width, const int height);

#endif