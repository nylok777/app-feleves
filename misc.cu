//
// Created by david on 5/17/26.
//
#include "misc.h"
#include "types.cuh"
#include <fstream>
#include <cuda/std/optional>
#include <iostream>
#include <sstream>
#include <numeric>


__host__ std::vector<std::pair<Double3, double>> read_csv(const std::string& filename, unsigned num_lines)
{
    using SolutionPair = std::pair<Double3, double>;

    std::vector<SolutionPair> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << '\n';
        return data;
    }

    std::string line;

    unsigned j = 0;

    while (std::getline(file, line) && j <= num_lines) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        int col = 0;

        std::getline(ss, token, ',');
        double time = std::stod(token);
        double sol[3];

        // Parse each comma-separated value
        while (std::getline(ss, token, ',') && col < 3) {
            sol[col++] = std::stod(token);
        }

        if (col == 3) {
            Double3 v{sol};
            data.emplace_back(v, time);
        }
        ++j;
    }

    return data;
}
