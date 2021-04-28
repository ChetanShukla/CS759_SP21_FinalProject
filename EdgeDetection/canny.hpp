#ifndef CANNY_HPP
#define CANNY_HPP

#include "point.hpp"
#include "global.hpp"
#include <unordered_map>

using namespace std;

void magnitude_matrix(double **pic, double **mag, double **x, double **y);

std::vector<Point*> peak_detection(double **mag, unordered_map<Point*, bool> peaks, double **x, double **y);

void recursiveDT(double **mag, double **final, unordered_map<Point*, bool> h, unordered_map<Point*, bool> peaks, int i, int j, int flag);

#endif