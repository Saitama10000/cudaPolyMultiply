#include <numeric>
#include <functional>
#include <memory>
#include <vector>
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fmt/color.h"
#include "glm/glm.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define err(ans) { errchk((ans), __FILE__, __LINE__); }
inline void errchk(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fmt::print(stderr, "GPUassert: {} {} {}\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <int poly_size>
__device__ void solve_poly(int i, float* a, float* b, float* inp)
{
    inp[i] = a[i];
    __syncthreads();

    float temp = 0.0f;
    for (int j = 0; j < poly_size; ++j)
        if (j <= i)
            temp += inp[j] * inp[i-j];
    
    b[i] = temp;
}

template <int poly_size>
__global__ void kernel(float* a, float* b, const int size)
{
    extern __shared__ float sm[];
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < size - size % poly_size; tid += gridDim.x * blockDim.x)
        solve_poly<poly_size>(
            tid % poly_size, 
            a + tid - tid % poly_size, 
            b + tid - tid % poly_size,
            sm + threadIdx.x - threadIdx.x % poly_size
        );
}

template <int poly_size>
std::vector<float> launch_gpu(std::vector<float> const& host_a)
{
    float* device_a; 
    float* device_b;
    err(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
    err(cudaMalloc(&device_b, host_a.size() * sizeof(float)));
    err(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));

    int BLOCKS;
    int THREADS;

    err(cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, kernel<poly_size>));
    fmt::print("kernel<<<{}, {}>>>\n", BLOCKS, THREADS);
    kernel<poly_size><<<BLOCKS, THREADS, THREADS * sizeof(float)>>>(device_a, device_b, host_a.size());
    err(cudaDeviceSynchronize());

    std::vector<float> host_b(host_a.size());
    err(cudaMemcpy(host_b.data(), device_b, host_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
    err(cudaFree(device_a));
    err(cudaFree(device_b));

    return host_b;
}

template <int poly_size>
std::vector<float> launch_cpu(std::vector<float> const& host_a)
{
    auto solve_poly = [](const float* a, float* b)
    {
        for (int i = 0; i < poly_size; ++i)
        {
            b[i] = 0.0;
            for (int k = 0; k <= i; ++k)
                b[i] += a[k] * a[i-k];
        }
    };

    std::vector<float> host_b(host_a.size());
    std::fill(host_b.begin(), host_b.end(), 0.0f);
    for (int i = 0; i < host_a.size() / poly_size; i++)
        solve_poly(&host_a[i * poly_size], &host_b[i * poly_size]);

    return host_b;
}

int main(int argc, char** argv)
{
    std::vector<float> host_a((1 << 23));
    std::iota(host_a.begin(), host_a.end(), 0);

    auto host_b = launch_cpu<32>(host_a);
    auto device_b = launch_gpu<32>(host_a);

    auto get_error = [](float a, float b)
    { 
        if (b == 0)
            return glm::abs(glm::abs(a)); 
        return glm::abs(glm::abs(a/b)-1); 
    };
    auto is_same = [&](float a, float b, float error = glm::pow(2.0f, -14.0f)){ return get_error(a, b) < error; };
    bool valid = std::equal(host_b.begin(),   host_b.end(), device_b.begin(), device_b.end(), is_same);
    fmt::print("Valid results: {}\n", valid);
}