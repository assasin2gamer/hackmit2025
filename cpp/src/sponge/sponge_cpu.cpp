#include "sponge.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

// ===================================================
// Correlation matrix
// ===================================================
void corr_update(const std::vector<float>& buffer, std::vector<float>& corr,
                 int N, int window) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float sum_x = 0, sum_y = 0, sum_xy = 0;
            float sum_x2 = 0, sum_y2 = 0;
            for (int w = 0; w < window; w++) {
                float xi = buffer[w * N + i];
                float yj = buffer[w * N + j];
                sum_x += xi; sum_y += yj;
                sum_xy += xi * yj;
                sum_x2 += xi * xi; sum_y2 += yj * yj;
            }
            float mean_x = sum_x / window;
            float mean_y = sum_y / window;
            float cov = (sum_xy / window) - mean_x * mean_y;
            float std_x = std::sqrt(sum_x2 / window - mean_x * mean_x);
            float std_y = std::sqrt(sum_y2 / window - mean_y * mean_y);
            float val = cov / (std_x * std_y + 1e-8f);
            corr[i * N + j] = corr[j * N + i] = val;
        }
    }
}

// ===================================================
// Split positive / negative adjacency
// ===================================================
void split_pos_neg(const std::vector<float>& corr,
                   std::vector<float>& A_pos,
                   std::vector<float>& A_neg,
                   int N, float eps) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = corr[i * N + j];
            if (val >= 0) {
                A_pos[i * N + j] = val;
                A_neg[i * N + j] = 0.0f;
            } else {
                A_pos[i * N + j] = 0.0f;
                A_neg[i * N + j] = -val;
            }
            if (i == j) {
                A_pos[i * N + j] += eps;
                A_neg[i * N + j] += eps;
            }
        }
    }
}

// ===================================================
// Degree and Laplacian
// ===================================================
void normalize_laplacian(const std::vector<float>& A,
                         std::vector<float>& L, int N) {
    std::vector<float> D(N, 0.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            D[i] += A[i * N + j];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dij = (i == j ? D[i] : 0.0f) - A[i * N + j];
            float norm_i = 1.0f / std::sqrt(D[i] + 1e-8f);
            float norm_j = 1.0f / std::sqrt(D[j] + 1e-8f);
            L[i * N + j] = norm_i * dij * norm_j;
        }
    }
}

// ===================================================
// Power iteration for eigenvectors
// ===================================================
void power_method(const std::vector<float>& mat, std::vector<float>& vecs,
                  int N, int k, int iters) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1, 1);

    for (int v = 0; v < k; v++) {
        std::vector<float> x(N);
        for (int i = 0; i < N; i++) x[i] = dist(gen);

        for (int it = 0; it < iters; it++) {
            std::vector<float> y(N, 0.0f);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    y[i] += mat[i * N + j] * x[j];
                }
            }
            float norm = 0.0f;
            for (float val : y) norm += val * val;
            norm = std::sqrt(norm);
            for (int i = 0; i < N; i++) x[i] = y[i] / (norm + 1e-8f);
        }
        for (int i = 0; i < N; i++) {
            vecs[v * N + i] = x[i];
        }
    }
}

// ===================================================
// Manual KMeans
// ===================================================
void kmeans_cpu(const std::vector<float>& X, int N, int dim, int k,
                int max_iter, std::vector<int>& labels) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<float> centroids(k * dim);
    for (int i = 0; i < k * dim; i++) centroids[i] = dist(gen);

    labels.assign(N, 0);

    for (int it = 0; it < max_iter; it++) {
        for (int i = 0; i < N; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            for (int c = 0; c < k; c++) {
                float dist2 = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = X[i * dim + d] - centroids[c * dim + d];
                    dist2 += diff * diff;
                }
                if (dist2 < best_dist) {
                    best_dist = dist2;
                    best_cluster = c;
                }
            }
            labels[i] = best_cluster;
        }

        std::vector<float> new_centroids(k * dim, 0.0f);
        std::vector<int> counts(k, 0);
        for (int i = 0; i < N; i++) {
            int c = labels[i];
            counts[c]++;
            for (int d = 0; d < dim; d++) {
                new_centroids[c * dim + d] += X[i * dim + d];
            }
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] /= counts[c];
                }
            }
        }
        centroids = new_centroids;
    }
}

// ===================================================
// Save to CSV
// ===================================================
// void save_to_csv(const std::string& filename,
//                  const std::vector<int>& labels,
//                  const std::vector<float>& embedding,
//                  int N, int dim) {
//     std::ofstream file(filename, std::ios::app);
//     for (int i = 0; i < N; i++) {
//         file << labels[i];
//         for (int d = 0; d < dim; d++) {
//             file << "," << embedding[i * dim + d];
//         }
//         file << "\n";
//     }
//     file.close();
// }

// ===================================================
// Main driver (CPU)
// ===================================================
void spongesym_live_cpu(int N, int window, int k, int ticks,
                        const std::string& out_file, bool test) {
    std::vector<float> buffer(window * N, 0.0f);
    std::vector<float> corr(N * N, 0.0f);
    std::vector<float> A_pos(N * N, 0.0f), A_neg(N * N, 0.0f);
    std::vector<float> L_pos(N * N, 0.0f), L_neg(N * N, 0.0f);
    std::vector<float> embedding(N * k, 0.0f);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0, 0.01);
    float eps = 1e-3f;

    for (int t = 0; t < ticks; t++) {
        int row = t % window;
        for (int i = 0; i < N; i++) {
            buffer[row * N + i] = dist(gen);
        }

        corr_update(buffer, corr, N, window);

        split_pos_neg(corr, A_pos, A_neg, N, eps);

        normalize_laplacian(A_pos, L_pos, N);
        normalize_laplacian(A_neg, L_neg, N);

        // Approximate generalized eigenproblem: inv(L_neg) * L_pos
        // For now just use L_pos as approximation
        power_method(L_pos, embedding, N, k, 20);

        std::vector<int> labels;
        kmeans_cpu(embedding, N, k, k, 10, labels);

        //if (!test) save_to_csv(out_file, labels, embedding, N, k);
    }
}
