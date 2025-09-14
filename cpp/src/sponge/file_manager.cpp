//
// Created by user on 9/13/25.
//

#include "sponge.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>

void save_to_csv(const std::string& filename,
                 const std::vector<int>& labels,
                 const std::vector<float>& embedding,
                 int N, int dim) {
    std::ofstream file(filename, std::ios::app);
    for (int i = 0; i < N; i++) {
        file << labels[i];
        for (int d = 0; d < dim; d++) {
            file << "," << embedding[i * dim + d];
        }
        file << "\n";
    }
    file.close();
}