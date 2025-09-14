#include "sponge.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void insert_new_returns(float* buffer, const float* new_row,
                                   int N, int window, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int row = t % window;
        buffer[row * N + idx] = new_row[idx];
    }
}

__global__ void update_sums(const float* new_row, const float* old_row,
                            float* sum, float* sum2, float* sum_xy,
                            int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xn = new_row[i];
        float xo = old_row[i];
        sum[i]  += xn - xo;
        sum2[i] += xn * xn - xo * xo;
        for (int j = 0; j < N; j++) {
            float yn = new_row[j];
            float yo = old_row[j];
            sum_xy[i * N + j] += xn * yn - xo * yo;
        }
    }
}

__global__ void corr_from_sums(const float* sum, const float* sum2,
                               const float* sum_xy, float* corr,
                               int N, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N && i <= j) {
        float mean_x = sum[i] / window;
        float mean_y = sum[j] / window;
        float cov   = (sum_xy[i * N + j] / window) - mean_x * mean_y;
        float var_x = (sum2[i] / window) - mean_x * mean_x;
        float var_y = (sum2[j] / window) - mean_y * mean_y;
        float denom = sqrtf(var_x * var_y + 1e-8f);
        float val   = cov / denom;
        corr[i * N + j] = corr[j * N + i] = val;
    }
}

__global__ void split_pos_neg(const float* corr, float* A_pos, float* A_neg,
                              int N, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float val = corr[i * N + j];
        if (val >= 0) { A_pos[i * N + j] = val; A_neg[i * N + j] = 0.0f; }
        else          { A_pos[i * N + j] = 0.0f; A_neg[i * N + j] = -val; }
        if (i == j) { A_pos[i * N + j] += eps; A_neg[i * N + j] += eps; }
    }
}

__global__ void row_sums(const float* A, float* D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += A[i * N + j];
        D[i] = sum;
    }
}

__global__ void normalize_laplacian(const float* A, const float* D, float* L, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float dij = (i == j ? D[i] : 0.0f) - A[i * N + j];
        float norm_i = 1.0f / sqrtf(D[i] + 1e-8f);
        float norm_j = 1.0f / sqrtf(D[j] + 1e-8f);
        L[i * N + j] = norm_i * dij * norm_j;
    }
}

void power_method(float* d_mat, float* d_vecs, int N, int k, int iters) {
    cublasHandle_t handle; cublasCreate(&handle);
    thrust::device_vector<float> d_tmp(N);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-1,1);
    for (int v=0; v<k; v++) {
        thrust::host_vector<float> h_x(N); for (int i=0;i<N;i++) h_x[i]=dist(gen);
        thrust::device_vector<float> d_x=h_x;
        for (int it=0; it<iters; it++) {
            float alpha=1.0f,beta=0.0f;
            cublasSgemv(handle,CUBLAS_OP_N,N,N,&alpha,d_mat,N,
                        thrust::raw_pointer_cast(d_x.data()),1,
                        &beta,thrust::raw_pointer_cast(d_tmp.data()),1);
            float norm; cublasSnrm2(handle,N,thrust::raw_pointer_cast(d_tmp.data()),1,&norm);
            float inv_norm=1.0f/(norm+1e-8f);
            cublasSscal(handle,N,&inv_norm,thrust::raw_pointer_cast(d_tmp.data()),1);
            d_x=d_tmp;
        }
        cudaMemcpy(d_vecs+v*N, thrust::raw_pointer_cast(d_x.data()), N*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cublasDestroy(handle);
}

__global__ void assign_clusters(const float* X, const float* centroids,
                                int* labels, int N, int k, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float best=1e30; int bestc=0;
        for (int c=0;c<k;c++) {
            float dist=0.0f;
            for (int d=0; d<dim; d++) {
                float diff=X[i*dim+d]-centroids[c*dim+d];
                dist+=diff*diff;
            }
            if(dist<best){best=dist;bestc=c;}
        }
        labels[i]=bestc;
    }
}

__global__ void update_centroids(const float* X, float* centroids,
                                 const int* labels, int N, int k, int dim) {
    int c=blockIdx.x; int d=threadIdx.x;
    if (c<k && d<dim) {
        float sum=0.0f; int count=0;
        for (int i=0;i<N;i++) if(labels[i]==c){sum+=X[i*dim+d]; count++;}
        if(count>0) centroids[c*dim+d]=sum/count;
    }
}

void kmeans_gpu(float* d_X,int N,int dim,int k,int max_iter,std::vector<int>& h_labels){
    thrust::device_vector<int> d_labels(N);
    thrust::device_vector<float> d_centroids(k*dim);
    thrust::host_vector<float> h_init(k*dim);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-1,1);
    for(int i=0;i<k*dim;i++) h_init[i]=dist(gen); d_centroids=h_init;
    for(int it=0;it<max_iter;it++){
        assign_clusters<<<(N+255)/256,256>>>(d_X,
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(d_labels.data()),N,k,dim);
        update_centroids<<<k,dim>>>(d_X,
            thrust::raw_pointer_cast(d_centroids.data()),
            thrust::raw_pointer_cast(d_labels.data()),N,k,dim);
    }
    h_labels.resize(N);
    cudaMemcpy(h_labels.data(), thrust::raw_pointer_cast(d_labels.data()), N*sizeof(int), cudaMemcpyDeviceToHost);
}

void spongesym_live_gpu(const std::string& input_folder,
                        const std::string& output_folder,
                        int window,int k,int ticks,bool test) {
    std::vector<std::string> instruments;
    std::vector<std::vector<std::string>> dates;
    std::vector<std::vector<float>> measures;

        for(auto& entry: fs::directory_iterator(input_folder)){
        if(entry.path().extension()==".csv"){
            std::ifstream f(entry.path()); std::string line;
            std::string inst=entry.path().stem().string();
            instruments.push_back(inst);
            std::vector<std::string> d; std::vector<float> m;
            std::getline(f,line);             while(std::getline(f,line)){
                std::stringstream ss(line); std::string tok;
                std::getline(ss,tok,','); d.push_back(tok);                 std::getline(ss,tok,','); float val=std::stof(tok); m.push_back(val);
            }
            dates.push_back(d); measures.push_back(m);
        }
    }
    int N=instruments.size();
    std::cout<<"Loaded "<<N<<" instruments\n";

    fs::create_directories(output_folder);
    std::vector<std::ofstream> files(N);
    for(int i=0;i<N;i++){
        std::string fname=output_folder+"/"+instruments[i]+".csv";
        files[i].open(fname);
        files[i]<<"date,label"; for(int d=0;d<k;d++) files[i]<<",eigvec"<<(d+1); files[i]<<"\n";
    }

        float *d_buffer,*d_corr,*A_pos,*A_neg,*D_pos,*D_neg,*L_pos,*L_neg,*M,*d_embedding;
    float *d_row,*d_old_row,*d_sum,*d_sum2,*d_sum_xy;
    CUDA_CHECK(cudaMalloc(&d_buffer,window*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_pos,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_neg,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_pos,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_neg,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&L_pos,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&L_neg,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&M,N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_embedding,N*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_old_row,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum2,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xy,N*N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum,0,N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum2,0,N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_xy,0,N*N*sizeof(float)));

    dim3 threads2D(16,16); dim3 blocks2D((N+15)/16,(N+15)/16);

        for(int t=0;t<ticks;t++){
        std::vector<float> row(N);
        for(int i=0;i<N;i++){
            if(t<measures[i].size()) row[i]=measures[i][t]; else row[i]=0.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_row,row.data(),N*sizeof(float),cudaMemcpyHostToDevice));

        int row_idx=t%window;
        CUDA_CHECK(cudaMemcpy(d_old_row,d_buffer+row_idx*N,N*sizeof(float),cudaMemcpyDeviceToDevice));
        insert_new_returns<<<(N+255)/256,256>>>(d_buffer,d_row,N,window,t);
        update_sums<<<(N+255)/256,256>>>(d_row,d_old_row,d_sum,d_sum2,d_sum_xy,N);
        corr_from_sums<<<blocks2D,threads2D>>>(d_sum,d_sum2,d_sum_xy,d_corr,N,window);
        split_pos_neg<<<blocks2D,threads2D>>>(d_corr,A_pos,A_neg,N,1e-3f);
        row_sums<<<(N+255)/256,256>>>(A_pos,D_pos,N);
        row_sums<<<(N+255)/256,256>>>(A_neg,D_neg,N);
        normalize_laplacian<<<blocks2D,threads2D>>>(A_pos,D_pos,L_pos,N);
        normalize_laplacian<<<blocks2D,threads2D>>>(A_neg,D_neg,L_neg,N);
        CUDA_CHECK(cudaMemcpy(M,L_pos,N*N*sizeof(float),cudaMemcpyDeviceToDevice));
        power_method(M,d_embedding,N,k,20);

        std::vector<int> labels; kmeans_gpu(d_embedding,N,k,k,10,labels);
        std::vector<float> h_embedding(N*k);
        CUDA_CHECK(cudaMemcpy(h_embedding.data(),d_embedding,N*k*sizeof(float),cudaMemcpyDeviceToHost));

        if(!test){
            for(int i=0;i<N;i++){
                std::string date=(t<dates[i].size()?dates[i][t]:std::to_string(t));
                files[i]<<date<<","<<labels[i];
                for(int d=0;d<k;d++) files[i]<<","<<h_embedding[i*k+d];
                files[i]<<"\n";
            }
        }
    }
    for(auto& f:files) f.close();

    cudaFree(d_buffer); cudaFree(d_corr); cudaFree(A_pos); cudaFree(A_neg);
    cudaFree(D_pos); cudaFree(D_neg); cudaFree(L_pos); cudaFree(L_neg);
    cudaFree(M); cudaFree(d_embedding); cudaFree(d_row); cudaFree(d_old_row);
    cudaFree(d_sum); cudaFree(d_sum2); cudaFree(d_sum_xy);
}
