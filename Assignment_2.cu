///////////////////////////|
//|File: Assignment_2.cu
//|Author: Jerrin C. Redmon
//|Language: C++
//|Version: 1.0.0
//|Date: April 24, 2025
///////////////////////////|

/*
 * Description:
 * CUDA program to compute the dot product of a 10x10 matrix M and a vector V.
 * Each row of the matrix is multiplied by the vector, and the results are stored in a result vector R.
 * The program verifies the results by comparing them with the expected output.
 * The expected output is the sum of the squares of the first 10 natural numbers.
 * The expected output is 385.
 * The program uses shared memory to store partial results for each row.
 * The program uses CUDA parallelization to speed up the computation.        
 */

//----------------------------------------------------------------


// Includes //
#include <iostream>                                                                                                 // For input/output
#include <cuda_runtime.h>                                                                                           // For CUDA runtime API

// Kernel function //
__global__ void matrix_vector_dot_product(int *matrix, int *vector, int *result, int size) {                       
    __shared__ int partials[10];                                                                                    // Shared memory to store partial results

    int row = blockIdx.x;                                                                                           // Get the row index
    int col = threadIdx.x;                                                                                          // Get the row and column indices


    if (col < size)                                                                                                 // Check bounds

        partials[col] = matrix[row * size + col] * vector[col];                                                     // Compute partial product
    else                                                                                                            // If out of bounds, set to 0
        partials[col] = 0;                                                                                          // Initialize to 0 if out of bounds
    __syncthreads();                                                                                                // Synchronize threads in the block

    if (col == 0) {                                                                                                 // Only one thread in each block will execute this

        int sum = 0;                                                                                                // Initialize sum

        for (int i = 0; i < size; ++i)                                                                              // Sum the partial products

            sum += partials[i];                                                                                     // Sum the partial products
        result[row] = sum;                                                                                          // Store the result in the result vector
    }
}

// Main //
int main() {

    const int size = 10;                                                                                            // Size of the matrix and vector
    int h_matrix[size * size], h_vector[size], h_result[size];                                                      // Host matrix, vector, and result vector

    for (int i = 0; i < size; ++i) {                                                                                // Initialize matrix and vector

        h_vector[i] = i + 1;                                                                                        // Initialize vector

        for (int j = 0; j < size; ++j)                                                                              // Initialize matrix

            h_matrix[i * size + j] = j + 1;                                                                         
    }

    int *d_matrix, *d_vector, *d_result;                                                                            // Allocate device memory
    cudaMalloc(&d_matrix, size * size * sizeof(int));                                                               // Allocate memory for matrix
    cudaMalloc(&d_vector, size * sizeof(int));                                                                      // Allocate memory for vector
    cudaMalloc(&d_result, size * sizeof(int));                                                                      // Allocate memory for result vector

    cudaMemcpy(d_matrix, h_matrix, size * size * sizeof(int), cudaMemcpyHostToDevice);                              // Copy matrix to device
    cudaMemcpy(d_vector, h_vector, size * sizeof(int), cudaMemcpyHostToDevice);                                     // Copy vector to device

    matrix_vector_dot_product<<<size, size>>>(d_matrix, d_vector, d_result, size);                                  // Launch kernel
    cudaDeviceSynchronize();                                                                                        // Synchronize device
    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);                                     // Copy result back to host

    std::cout << "Result:\n";                                                                                       // Print result vector

    for (int i = 0; i < size; ++i)                                                                                 
    
        std::cout << h_result[i] << " ";                                                                            
    std::cout << "\n";                                                                                              

    int expected = 0;                                                                                               // Expected value

    for (int j = 0; j < size; ++j)                                                                                  // Calculate expected value

        expected += (j + 1) * (j + 1);                                                                              

    bool correct = true;                                                                                            // Flag to check if the result is correct

    for (int i = 0; i < size; ++i) {                                                                                // Check if the result matches the expected value

        if (h_result[i] != expected) {                                                                              // Check if the result matches the expected value
            
            std::cout << "Mismatch at R[" << i << "]: expected " << expected << ", got " << h_result[i] << "\n";    // Print mismatch
            correct = false;
        }
    }

    if (correct)
        std::cout << "Verification Passed.\n";
    else
        std::cout << "Verification Failed.\n";

    cudaFree(d_matrix);                                                                                             // Free device memory
    cudaFree(d_vector);                                                                                             // Free device memory
    cudaFree(d_result);                                                                                             // Free device memory

    return 0;
}
