//
// Created by david on 5/17/26.
//

#ifndef APP_FELEVES_MISC_H
#define APP_FELEVES_MISC_H
#include <string>
#include <vector>

#include "types.cuh"

std::vector<std::pair<Double3, double>> read_csv(const std::string& filename);

#endif //APP_FELEVES_MISC_H
