///////////////////////////|
//|File: Assignment_1.cu
//|Author: Jerrin C. Redmon
//|Language: C++
//|Version: 1.0.0
//|Date: April 24, 2025
///////////////////////////|

/*
 * Description:
 * CUDA program to multiply each element M[i][j] of a 10x10 matrix M
 * by a corresponding vector element V[j]. This is not a matrix-vector dot product,
 * but a column-wise multiplication across the matrix using CUDA parallelization. 
 * The program verifies the results by comparing them with the expected output.         
 */

//----------------------------------------------------------------


// Includes //
#include <iostream>                                                                                         // For input/output
#include <cuda_runtime.h>                                                                                   // For CUDA runtime API

// Kernal function //
__global__ void mat_multiply(int *matrix, int *vector, int size) {                                          

    int row = blockIdx.x;                                                                                   // Get the row index
    int col = threadIdx.x;                                                                                  // Get the row and column indices

    if (row < size && col < size) {                                                                         // Check bounds

        int idx = row * size + col;                                                                         // Calculate the index for the matrix
        matrix[idx] *= vector[col];                                                                         // Element-wise multiplication
    }
}

// Main //
int main() {

    const int size = 10;                                                                                    // Size of the matrix and vector
    const int matrix_size = size * size;                                                                    // Size of the matrix
    int h_matrix[matrix_size], h_vector[size];                                                              // Host matrix and vector

 
    for (int i = 0; i < size; ++i) {                                                                        // Initialize matrix and vector

        h_vector[i] = i + 1;                                                                                // Initialize vector

        for (int j = 0; j < size; ++j) {                                                                    // Initialize matrix

            h_matrix[i * size + j] = j + 1;                                                                 
        }
    }

    int *d_matrix, *d_vector;                                                                               // Allocate device pointers
    cudaMalloc(&d_matrix, matrix_size * sizeof(int));                                                       // Allocate memory for matrix on device
    cudaMalloc(&d_vector, size * sizeof(int));                                                              // Allocate memory for matrix and vector on device

    cudaMemcpy(d_matrix, h_matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);                      // Copy matrix to device
    cudaMemcpy(d_vector, h_vector, size * sizeof(int), cudaMemcpyHostToDevice);                             // Copy vector to device

    mat_multiply<<<size, size>>>(d_matrix, d_vector, size);                                                 // Launch kernel with 10 blocks and 10 threads
    cudaDeviceSynchronize();                                                                                // Synchronize to ensure kernel execution is complete
    cudaMemcpy(h_matrix, d_matrix, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);                      // Copy result back to host

    std::cout << "Matrix Results:\n";                                                                       // Print modified matrix

    for (int i = 0; i < size; ++i) {                                                                        // Loop through each row
        
        for (int j = 0; j < size; ++j)                                                                      // Print each row

            std::cout << h_matrix[i * size + j] << " ";                                                     // Print each element
        std::cout << "\n";
    }

    // Verification //
    std::cout << "\nVerifying Results...\n";
    bool correct = true;                                                                                    // Flag to check correctness

    for (int i = 0; i < size; ++i) {                                                                        // Loop through each row

        for (int j = 0; j < size; ++j) {                                                                    // Loop through each column
            
            int expected = (j + 1) * (j + 1);                                                               // since M[i][j] = j+1 and V[j] = j+1
            if (h_matrix[i * size + j] != expected) {                                                       // Check if the result is as expected

                std::cout << "Mismatch at M[" << i << "][" << j << "]: expected " << expected               
                          << ", got " << h_matrix[i * size + j] << "\n";                                    // Print mismatch
                correct = false;
            }
        }
    }

    if (correct)
        std::cout << "Verification Passed: All matrix elements are correct.\n";                             // Print verification result
    else
        std::cout << "Verification Failed: One or more elements are incorrect.\n";                     

    cudaFree(d_matrix);                                                                                     // Free device memory
    cudaFree(d_vector);                                                                                     // Free device memory    

    return 0;
}
