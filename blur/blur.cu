// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image.h"
#include "util/stb_image_write.h"

__device__
int get_rowmajor_index(int row, int col, int channel, int width, int channels) {
    return row * (width * channels) + col * channels + channel;
}

__global__
void blurKernel(const float* input_image_d, float* output_image_d, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    // thread does stuff only if it is on a valid pixel
    if (row < height && col < width) {

        float surrounding_sum = 0;
        int num_surrounding = 0;

        for ( int dx = -1; dx <= 1; ++dx ) {
            for ( int dy = -1; dy <= 1; ++dy ) {
                int new_row = row + dy;
                int new_col = col + dx;
                if (new_row >= 0 && new_row < height && new_col >= 0 && new_col < width) {
                    surrounding_sum += input_image_d[get_rowmajor_index(new_row, new_col, channel, width, 3)];
                    num_surrounding += 1;
                }
            }
        }
        output_image_d[get_rowmajor_index(row, col, channel, width, 3)] = surrounding_sum /= num_surrounding;
    }

}

void blur(const float* input_image, int width, int height, float* output_image) {
    int image_size = 3 * width * height * sizeof(float);

    // allocate room for both the input image and output image on the GPU
    float* input_image_d;
    float* output_image_d;
    cudaMalloc((void**) &input_image_d, image_size);
    cudaMalloc((void**) &output_image_d, image_size);
    // copy the input image onto the GPU pointer
    cudaMemcpy(input_image_d, input_image, image_size, cudaMemcpyHostToDevice);

    // call kernel
    // image is a 3D array of size 3 x width x height that has been flattened
    // lets make each block have 16x16 threads
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(width / (float)blockDim.x), ceil(height / (float)blockDim.y), 3);

    // call the kernel
    blurKernel<<<gridDim, blockDim>>>(input_image_d, output_image_d, width, height);

    // the kernel directly operates on output_image_d. So we just copy the result into output_image
    cudaMemcpy(output_image, output_image_d, image_size, cudaMemcpyDeviceToHost);
    // then we just need to free the two cuda dynamically allocated arrays we made
    cudaFree(input_image_d);
    cudaFree(output_image_d);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.png> <output.png>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }

    float* input = new float[width * height * 3];
    float* output = new float[width * height * 3];

    for (int i = 0; i < width * height * 3; ++i)
        input[i] = img[i] / 255.0f;

    blur(input, width, height, output);

    for (int i = 0; i < width * height * 3; ++i)
        img[i] = (unsigned char)(output[i] * 255.0f);

    const char* ext = strrchr(argv[2], '.');
    if (ext && strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)
        stbi_write_jpg(argv[2], width, height, 3, img, 90);
    else
        stbi_write_png(argv[2], width, height, 3, img, width * 3);

    stbi_image_free(img);
    delete[] input;
    delete[] output;

    return 0;
}
