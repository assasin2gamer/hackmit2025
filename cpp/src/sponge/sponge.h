#ifndef SPONGE_HPP
#define SPONGE_HPP

#include <string>
#include <vector>

// =======================
// Public API Declarations
// =======================

// Run live SPONGEsym clustering on streaming data
// N        = number of assets
// window   = rolling window size
// k        = number of clusters
// ticks    = number of iterations (simulated feed length)
// out_file = CSV file for labels + embeddings
void save_to_csv(const std::string& filename,
                 const std::vector<int>& labels,
                 const std::vector<float>& embedding,
                 int N, int dim);
void spongesym_live_cpu(int N, int window, int k, int ticks,
                    const std::string& out_file, bool test);
// void spongesym_live_gpu(int N, int window, int k, int ticks,
//                     const std::string& out_file, bool test);
void spongesym_live_gpu(const std::string& input_folder,
                        const std::string& output_folder,
                        int window, int k, int ticks,
                        bool test);

#endif // SPONGE_HPP