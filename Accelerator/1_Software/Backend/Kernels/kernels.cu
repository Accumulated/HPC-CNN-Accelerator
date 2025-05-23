#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "CommonInclude.h"

#define WARP_SIZE 32

/* Kernel definitions */
__global__ void INPUT_UNROLLING(int stride, int Filter_Height,
                                float *Input, int H1, int W1, int D1,
                                float *X_unrolled, int H2, int W2, int D2,
                                int Output_Height, int Output_Width)
{
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Select row and column values
    int row =  by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int depth = bz;

    int col_no_strided = col, row_no_strided = row;
    int depth_offset = depth * W2 * Filter_Height * Filter_Height;

    /*
      Note for bx, by and bz= 0, stride = 2:
          @ tx = 0, ty = 0 -> First multiply the col * stride, row * stride; = 0, 0
                            you are shifting in x direction using local col
                            you are shifting in y direction using local row;
          @ tx = 1, ty = 0 -> First multiply the col * stride, row * stride; = 2, 0
                            you are shifting in x direction using local col
                            you are shifting in y direction using local row;
          @ tx = 0, ty = 1 -> First multiply the col * stride, row * stride; = 0, 2
                            you are shifting in x direction using local col
                            you are shifting in y direction using local row;
    */

    col *= stride; row *= stride;

    // Limit number of threads
    if (row_no_strided < Output_Height && col_no_strided < Output_Width && depth < D1)
    {

      // Each thread unrolls k x k elements
      for (int local_row = 0; local_row < Filter_Height; local_row++){

        for (int local_col = 0; local_col < Filter_Height; local_col++){

          // 1. local row and column shifts affect the locations in Unrolled matrix
          // 2. For each col and row non strided values -> you are adding an offset to columns and rows in Unrolled matrix
          // 3. Offset the depth using "depth_offset" variable
          X_unrolled[local_col * W2 + local_row * Filter_Height * W2 + col_no_strided + row_no_strided * Output_Width + depth_offset] =
          Input[(row + local_row) * W1 + (col + local_col) + depth * H1 * W1];

        }

      }

    }

}

__global__ void DWConv2d_kernel(float *Input, int H1, int W1, int D1,
                                float *Filter, int H2, int W2, int D2,
                                float *Output, int H3, int W3, int D3,
                                int stride)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int dep = bz;

    float Pvalue = 0;

    if (row < H3 && col < W3 && dep < D3)
    {
      // 1 thread unrolls kxk section
      for (int j = 0; j < H2; j++)
      {
        for (int i = 0; i < W2; i++)
        {
            Pvalue += Filter[j * W2 + i + dep * H2 * W2] *
                Input[(j * W1 + row * stride * W1) + (i + col * stride) + dep * H1 * W1];
        }
      }
      Output[row * W3 + col + dep * H3 * W3] = Pvalue;
    }

}

__global__ void MatrixMulKernel(float *M, int H1, int W1, int D1,
                                float *N, int H2, int W2, int D2,
                                float *P, int H3, int W3, int D3,
                                int num_blocks, int activation,
                                int IS_BIASED, float *bias_mat)
{
  __shared__ float Mds[Tile_GEMM][Tile_GEMM];
  __shared__ float Nds[Tile_GEMM][THREAD_GRANULARITY_BLOCKS * Tile_GEMM];

  int bx = blockIdx.x * THREAD_GRANULARITY_BLOCKS;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the d_P element to work on
  int Row = by * Tile_GEMM + ty;
  int Col = bx * Tile_GEMM + tx;
  float Pvalue = 0;
  float Pvalue_2 = 0;

  // Loop over the d_M and d_N tiles required to compute d_P element
  for (int ph = 0; ph < num_blocks; ++ph)
  {
    // Collaborative loading of d_M and d_N tiles into shared memory
    if ((Row < H1) && (ph * Tile_GEMM + tx) < W1)
    {
      Mds[ty][tx] = M[Row * W1 + ph * Tile_GEMM + tx];
    }

    if ((ph * Tile_GEMM + ty) < H2 && Col < W2)
    {
      Nds[ty][tx] = N[(ph * Tile_GEMM + ty) * W2 + Col];
    }

    if ((ph * Tile_GEMM + ty) < H2 && Col + Tile_GEMM < W2)
    {
      Nds[ty][tx + Tile_GEMM] = N[(ph * Tile_GEMM + ty) * W2 + Col + Tile_GEMM];
    }

    __syncthreads();

    for (int k = 0; k < Tile_GEMM && (ph * Tile_GEMM) + k < W1; ++k)
    {
      Pvalue += Mds[ty][k] * Nds[k][tx];
      if (Col + Tile_GEMM < W2)
        Pvalue_2 += Mds[ty][k] * Nds[k][tx + Tile_GEMM];
    }

    __syncthreads();

  }

  if ((Row < H1) && (Col < W2))
  {
    P[Row * W3 + Col] = Pvalue;

    switch (IS_BIASED)
    {
      case BIASED:
        Pvalue = Pvalue + bias_mat[Row];
        break;

      default:
        break;
    }

    switch (activation)
    {
      case SWISH_ACTIVATION:
        // Swish activation function
        P[Row * W3 + Col] = Pvalue / (1.0f + expf(-1.0f * Pvalue));
        break;

      case SIGMOID_ACTIVATION:
        // Sigmoid activation function
        P[Row * W3 + Col] = 1.0f / (1.0f + expf(-1.0f * Pvalue));
        break;

      default:
        break;
    }
  }

  if ((Row < H1) && (Col + Tile_GEMM < W2))
    {
      P[Row * W3 + Col + Tile_GEMM] = Pvalue_2;

      switch (IS_BIASED)
      {
        case BIASED:
          Pvalue_2 = Pvalue_2 + bias_mat[Row];
          break;

        default:
          break;
      }

      switch (activation)
      {
        case SWISH_ACTIVATION:
          // Swish activation function
          P[Row * W3 + Col + Tile_GEMM] = Pvalue_2 / (1.0f + expf(-1.0f * Pvalue_2));
          break;

        case SIGMOID_ACTIVATION:
          // Sigmoid activation function
          P[Row * W3 + Col + Tile_GEMM] = 1.0f / (1.0f + expf(-1.0f * Pvalue_2));
          break;

        default:
          break;
      }
    }

}


__global__ void ConvChannelElementWiseMultiplication(float *A, int H1, int W1, int D1,
                                                     float *B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z;

    int index = depth * W1 * H1 + row * W1 + col;

    if ((row < H1) && (col < W1) && (depth < D1))
    {
        A[index] = A[index] * B[depth];
    }
}

__global__ void CastingDivision(float *A, int W1, float B)
{
    // Warning: 1-D kernel only in x dir.

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((col < W1))
    {
        A[col] /= B;
    }
}


// Used with MBConv layers that has skip identity = true
__global__ void Identity_Skip(float *A,  int H1, int W1, int D1,
                              float *B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z;

    int index = depth * W1 * H1 + row * W1 + col;

    if ((row < H1) && (col < W1) && (depth < D1))
    {
        A[index] = A[index] + B[index];
    }
}


__global__ void MatrixAddKernel(float* A, float* B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
      A[idx] += B[idx];
  }
}


__global__ void Complete_Padding_Process(float *output, int outputHeight, int outputWidth, int outputDepth,
                                         float *input, int inputHeight, int inputWidth, int inputDepth,
                                         int paddingValue) {

    // Calculate the row and column indices for the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the current thread is within the bounds of the output matrix
    if (row < outputHeight && col < outputWidth && depth < outputDepth) {
        // Determine the corresponding position in the input matrix
        int inputRow = row - paddingValue;
        int inputCol = col - paddingValue;

        // Check if the current position is within the bounds of the input matrix
        if (inputRow >= 0 && inputRow < inputHeight &&
            inputCol >= 0 && inputCol < inputWidth) {

            // Copy the value from the input matrix to the output matrix
            output[depth * outputWidth * outputHeight + row * outputWidth + col] =
                  input[depth * inputWidth * inputHeight + inputRow * inputWidth + inputCol];

        } else {
            // Set the padding value (e.g., zero) for out-of-bounds positions
            output[depth * outputWidth * outputHeight + row * outputWidth + col] = 0.0f;
        }
    }
}


/* Batch Normalization Kernels */
const int BLOCK_SIZE = 128;

__global__ void BN_Kernel_Mean_Reduction(float *input, int H1, int W1, int D1,
                                         float *Mean, int W2)
{
    /*
        This code works on 2 * Block_Size elements.
        i.e. for 512 Block_Size -> we are reducing 1024 elements.
        Each thread loads 2 elements, one at tx and the
        other shifted by blockIdx.x.
    */

    __shared__ float partialSum[2 * BLOCK_SIZE];
    float tmp = 0;

    int tx = threadIdx.x;
    int bx = blockDim.x;

    int by_index = blockIdx.y;
    int bx_index = blockIdx.x;

    // The start variable is to get offset for input matrix in loading
    int start = blockIdx.x * (2 * blockDim.x);
    int start_yDir = blockIdx.y * W1;

    if (start + tx < W1 && start_yDir < H1 * W1)
        // Load 2 elements in the shared memory
        partialSum[tx] = input[start + tx + start_yDir];
    else
        partialSum[tx] = tmp;

    if (tx + bx + start < W1 && start_yDir < H1 * W1)
        partialSum[tx] += input[start + bx + tx + start_yDir];
    else
        partialSum[tx] += tmp;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride = stride / 2.0f)
    {
        if (tx < stride)
            partialSum[tx] += partialSum[tx + stride];
        __syncthreads();
    }

    // Reduction tree with shuffle instructions
    float sum = 0;
    if(tx < WARP_SIZE)
    {
      sum = partialSum[tx] + partialSum[tx + WARP_SIZE];
      for(unsigned int stride = WARP_SIZE/2; stride > 0; stride /= 2)
      {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
      }
    }
    if (tx == 0)
        Mean[bx_index + by_index * W2] = sum;

}

__global__ void ElementWiseSquaring(float *A, int H1, int W1, int D1)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z;

    int index = depth * W1 * H1 + row * W1 + col;

    if ((row < H1) && (col < W1) && (depth < D1))
    {
        A[index] = A[index] * A[index];
    }
}

__global__ void ElementWiseSubtraction(float *A, int H1, int W1, int D1,
                                       float *mean)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x  + threadIdx.x;
    int depth = blockIdx.z;

    int index = depth * W1 * H1 + row * W1 + col;

    if ((row < H1) && (col < W1) && (depth < D1))
    {
        A[index] = A[index] - mean[depth];
    }
}


__global__ void BatchNormKernel(float *InputMatrixElements, float *OutputMatrixElements,
                                      int H1, int W1, int D1,
                                      float *D_mean, float *D_variance,
                                      float *D_weight, float *D_bias,
                                      int activate)
{
    // Activate values are assigned as follow
    /*
      0 -> no activation, 1 -> swish, 2 -> sigmoid
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z;

    int index = depth * W1 * H1 + row * W1 + col;
    int index3 = depth;

    if ((row < H1) && (col < W1) && (depth < D1))
    {
        OutputMatrixElements[index] = ((InputMatrixElements[index] - D_mean[index3]) / (sqrtf(D_variance[index3] + 0.001f))) * D_weight[index3]
                                      + D_bias[index3];

        switch (activate) {

          case SWISH_ACTIVATION:
              // Swish activation function
              OutputMatrixElements[index] = OutputMatrixElements[index] / (1.0f + expf(-1.0f * OutputMatrixElements[index]));
              break;

          case SIGMOID_ACTIVATION:
              // Sigmoid activation function
              OutputMatrixElements[index] = 1.0f / (1.0f + expf(-1.0f * OutputMatrixElements[index]));
              break;

          default:
              break;
        }
    }
}