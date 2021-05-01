#ifndef CANNY_HPP
#define CANNY_HPP

#include "point.hpp"
#include "global.hpp"
#include <unordered_map>

using namespace std;

std::vector<Point*> peak_detection(double **mag, unordered_map<Point*, bool> peaks, double **x, double **y);

void recursiveDoubleThresholding(double **mag, double **final, unordered_map<Point*, bool> visited, 
                                    unordered_map<Point*, bool> peaks, int i, int j, int flag);

#endif