%%cu
#include <stdio.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

    using namespace std;

__global__ void matproduct(int *l, int *m, int *n, int row1, int col1, int row2, int col2)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    int i;
    n[(col2 * row1 * z) + (col2 * y) + x] = 0;

    for (i = 0; i < col1; i++)
    {
        n[(col2 * row1 * z) + (col2 * y) + x] += l[(col1 * row1 * z) + (col1 * y) + i] * m[(col2 * row2 * z) + (col2 * i) + x];
    }
}

int main()
{
    int num_matrices; // 1000
    int row1;         // 25
    int col1;         // 25
    int row2;         // 25
    int col2;         // 25

    ifstream file("input.txt");
    if (!file.is_open())
    {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        int k, m, n, p;

        if (iss >> k >> m >> n >> p)
        {
            std::cout << "K: " << k << ", M: " << m << ", N: " << n << ", p: " << p << std::endl;
            num_matrices = k;
            row1 = m;
            col1 = n;
            row2 = n;
            col2 = p;
        }
        else
        {
            std::cerr << "Failed to k, m, n, p from the line." << std::endl;
        }
    }

    int a[row1 * col1 * num_matrices];
    int b[row2 * col2 * num_matrices];
    int c[row1 * col2 * num_matrices];

    int *d, *e, *f;
    int i, j;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // Initialize matrices a and b
    int element = 1;
    for (int k = 0; k < num_matrices; k++)
    {
        for (i = 0; i < row1; i++)
        {
            for (j = 0; j < col1; j++)
            {
                a[(col1 * row1 * k) + (col1 * i) + j] = element++;
            }
        }

        element = 2;
        for (i = 0; i < row2; i++)
        {
            for (j = 0; j < col2; j++)
            {
                b[(col2 * row2 * k) + (col2 * i) + j] = element++;
            }
        }
    }

    cudaMalloc((void **)&d, row1 * col1 * num_matrices * sizeof(int));
    cudaMalloc((void **)&e, row2 * col2 * num_matrices * sizeof(int));
    cudaMalloc((void **)&f, row1 * col2 * num_matrices * sizeof(int));

    cudaMemcpy(d, a, row1 * col1 * num_matrices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, b, row2 * col2 * num_matrices * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(col2, row1, num_matrices);

    matproduct<<<grid, 1>>>(d, e, f, row1, col1, row2, col2);

    cudaMemcpy(c, f, row1 * col2 * num_matrices * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nProduct of matrices:\n");
    for (int k = 0; k < 1; k++)
    {
        printf("Matrix %d:\n", k + 1);
        for (i = 0; i < row1; i++)
        {
            for (j = 0; j < col2; j++)
            {
                printf("%d\t", c[(col2 * row1 * k) + (col2 * i) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("\nTime taken: %f\n ", milliseconds);

    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}
