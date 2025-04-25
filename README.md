# Nvidia_CUDA_Assignment

Author: **Jerrin C. Redmon**  
Date: **April 24, 2025**

This project contains two CUDA programs designed to demonstrate matrix and vector operations in parallel using GPU acceleration.

---

##  Assignment 1: Column-wise Matrix × Vector Multiplication

### 🔹 Description
This CUDA program multiplies each element `M[i][j]` of a 10×10 matrix `M` by the corresponding element in a vector `V[j]`. This is **not** a dot product. Instead, it **scales each column** of the matrix independently by the vector entry at that column index.

### 🔹 Kernel Details
- **Threads per Block**: 10 (one per column)
- **Blocks**: 10 (one per row)
- **Operation**:
  ```cpp
  matrix[i][j] = matrix[i][j] * vector[j];
  ```
---

##  Assignment 2: Row-wise Dot Product (Matrix × Vector)

### 🔹 Description
This program computes the dot product of each **row** in a 10×10 matrix `M` with a 10-element vector `V`, producing a result vector `R` of size 10 where:
```cpp
R[i] = dot(M[i], V) = Σ (M[i][j] * V[j])
```
All rows are identical and the result is `R[i] = 385`.

### 🔹 Kernel Details
- **Threads per Block**: 10 (one per element in row)
- **Blocks**: 10 (one per row)
- **Uses Shared Memory**: Yes
- **Uses Synchronization**: Yes (`__syncthreads()` for partial sum)


---

## ✅ Output Verification

- **Assignment 1**: Result matrix where each `M[i][j] = original M[i][j] * V[j]`.
- **Assignment 2**: Result vector `R[i] = 385` for all `i`, verified against expected dot product sum.

---

