// aktualnie testowana wersja kernela
#define TEST_FUNC matrixMulCUDA_v0

// System includes
#define WIN32
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// wersja oryginalna bez zrównoleglenia pobierania i obliczeń
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA_v0(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// równoległe obliczenia i pobieranie do rejestrów (obliczenia jako pierwsze w kodzie)
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA_v1(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    register float localA = A[aBegin + wA * ty + tx];
    register float localB = B[bBegin + wB * ty + tx];

    for (int a = aBegin + aStep, b = bBegin + bStep;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        As[ty*BLOCK_SIZE + tx] = localA;
        Bs[ty*BLOCK_SIZE + tx] = localB;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + tx];
        }

        localA = A[a + wA * ty + tx];
        localB = B[b + wB * ty + tx];

        __syncthreads();
    }

    // last loop was fetched but not processed
    // so we need to do it now
    As[ty*BLOCK_SIZE + tx] = localA;
    Bs[ty*BLOCK_SIZE + tx] = localB;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub += As[ty*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// równoległe obliczenia i pobieranie do rejestrów (pobieranie jako pierwsze w kodzie)
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA_v2(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    register float localA = A[aBegin + wA * ty + tx];
    register float localB = B[bBegin + wB * ty + tx];

    for (int a = aBegin + aStep, b = bBegin + bStep;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        As[ty*BLOCK_SIZE + tx] = localA;
        Bs[ty*BLOCK_SIZE + tx] = localB;
        __syncthreads();

        localA = A[a + wA * ty + tx];
        localB = B[b + wB * ty + tx];

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + tx];
        }

        __syncthreads();
    }

    // last loop was fetched but not processed
    // so we need to do it now
    As[ty*BLOCK_SIZE + tx] = localA;
    Bs[ty*BLOCK_SIZE + tx] = localB;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub += As[ty*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// równoległe obliczenia i pobieranie do pamięci współdzielonej (obliczenia jako pierwsze w kodzie)
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA_v3(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    __shared__ float As1[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE * BLOCK_SIZE];

    __shared__ float As2[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE * BLOCK_SIZE];

    float *currentA = As1;
    float *currentB = Bs1;
    float *nextA = As2;
    float *nextB = Bs2;

    currentA[ty*BLOCK_SIZE + tx] = A[aBegin + wA * ty + tx];
    currentB[ty*BLOCK_SIZE + tx] = B[bBegin + wB * ty + tx];

    __syncthreads();

    for (int a = aBegin + aStep, b = bBegin + bStep;
         a <= aEnd;
         a += aStep, b += bStep)
    {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += currentA[ty*BLOCK_SIZE + k] * currentB[k*BLOCK_SIZE + tx];
        }

        nextA[ty*BLOCK_SIZE + tx] = A[a + wA * ty + tx];
        nextB[ty*BLOCK_SIZE + tx] = B[b + wB * ty + tx];

        // swap buffers
        auto tmp = currentA;
        currentA = nextA;
        nextA = tmp;

        tmp = currentB;
        currentB = nextB;
        nextB = tmp;

        __syncthreads();
    }

    // last loop was fetched but not processed
    // so we need to do it now
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub += currentA[ty*BLOCK_SIZE + k] * currentB[k*BLOCK_SIZE + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// równoległe obliczenia i pobieranie do pamięci współdzielonej (pobieranie jako pierwsze w kodzie)
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA_v4(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;
    float Csub = 0;

    __shared__ float As1[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs1[BLOCK_SIZE * BLOCK_SIZE];

    __shared__ float As2[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs2[BLOCK_SIZE * BLOCK_SIZE];

    float *currentA = As1;
    float *currentB = Bs1;
    float *nextA = As2;
    float *nextB = Bs2;

    currentA[ty*BLOCK_SIZE + tx] = A[aBegin + wA * ty + tx];
    currentB[ty*BLOCK_SIZE + tx] = B[bBegin + wB * ty + tx];

    __syncthreads();

    for (int a = aBegin + aStep, b = bBegin + bStep;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        nextA[ty*BLOCK_SIZE + tx] = A[a + wA * ty + tx];
        nextB[ty*BLOCK_SIZE + tx] = B[b + wB * ty + tx];

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += currentA[ty*BLOCK_SIZE + k] * currentB[k*BLOCK_SIZE + tx];
        }

        // swap buffers
        auto tmp = currentA;
        currentA = nextA;
        nextA = tmp;

        tmp = currentB;
        currentB = nextB;
        nextB = tmp;

        __syncthreads();
    }

    // last loop was fetched but not processed
    // so we need to do it now
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub += currentA[ty*BLOCK_SIZE + k] * currentB[k*BLOCK_SIZE + tx];
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

#define assertSuccess(error) do {\
    cudaError_t e = (error);\
    if (e != cudaSuccess) {\
        fprintf(stderr, "%s:%d:%s Error: %s\n", __FILE__, __LINE__, __func__, cudaGetErrorString(e));\
        exit(EXIT_FAILURE);\
    }\
} while (0)

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);
    assertSuccess(error);

    error = cudaMalloc((void **) &d_B, mem_size_B);
    assertSuccess(error);

    error = cudaMalloc((void **) &d_C, mem_size_C);
    assertSuccess(error);

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    assertSuccess(error);

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    assertSuccess(error);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
#if 0
    TEST_FUNC<8><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
#else
    if (block_size == 16) {
        TEST_FUNC<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        TEST_FUNC<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
#endif

    cudaDeviceSynchronize();
    printf("done\n");

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);
    assertSuccess(error);

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);
    assertSuccess(error);

    // Record the start event
    error = cudaEventRecord(start, NULL);
    assertSuccess(error);

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
#if 0
        TEST_FUNC<8><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
#else
        if (block_size == 16) {
            TEST_FUNC<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else {
            TEST_FUNC<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
#endif
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);
    assertSuccess(error);

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);
    assertSuccess(error);

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    assertSuccess(error);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    assertSuccess(error);

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
        float epsilon = 1e-3;
        if (fabs(h_C[i] - (dimsA.x * valB)) > epsilon) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %g\n", i, h_C[i], dimsA.x*valB, epsilon);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
if (correct) {
        return EXIT_SUCCESS; } else {
        return EXIT_FAILURE;
    }
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    assertSuccess(error);

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}
