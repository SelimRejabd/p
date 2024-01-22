#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char const *argv[])
{
    int m, n, p, count;
    int l_count;
    int size, id;
    double t_start, t_end;

    int *l_a;
    int *l_b;
    int *l_c;

    int *mat_a;
    int *mat_b;
    int *mat_c;

    MPI_Init(NULL, NULL);
    MPI_Comm Comm = MPI_COMM_WORLD;
    MPI_Status status;
    MPI_Comm_rank(Comm, &id);
    MPI_Comm_size(Comm, &size);

    if (id == 0)
    {
        // Input data for matrix multiplication
        scanf("%d", &count);
        scanf("%d %d %d", &m, &n, &p);

        mat_a = (int *)malloc(count * m * n * sizeof(int));
        mat_b = (int *)malloc(count * n * p * sizeof(int));
        mat_c = (int *)malloc(count * m * p * sizeof(int));

        // Initialize matrices mat_a and mat_b
        for (int i = 0; i < count * m * n; i++)
            mat_a[i] = 1;

        for (int i = 0; i < count * n * p; i++)
            mat_b[i] = 1;

        // Send matrix dimensions to other processes
        for (int i = 1; i < size; i++)
        {
            MPI_Send(&count, 1, MPI_INT, i, 0, Comm);
            MPI_Send(&m, 1, MPI_INT, i, 1, Comm);
            MPI_Send(&n, 1, MPI_INT, i, 2, Comm);
            MPI_Send(&p, 1, MPI_INT, i, 3, Comm);
        }
    }
    else
    {
        // Receive matrix dimensions
        MPI_Recv(&count, 1, MPI_INT, 0, 0, Comm, &status);
        MPI_Recv(&m, 1, MPI_INT, 0, 1, Comm, &status);
        MPI_Recv(&n, 1, MPI_INT, 0, 2, Comm, &status);
        MPI_Recv(&p, 1, MPI_INT, 0, 3, Comm, &status);
    }

    l_count = count / size;
    l_a = (int *)malloc(l_count * m * n * sizeof(int));
    l_b = (int *)malloc(l_count * n * p * sizeof(int));
    l_c = (int *)malloc(l_count * m * p * sizeof(int));

    // Scatter matrices mat_a and mat_b
    MPI_Scatter(mat_a, l_count * m * n, MPI_INT, l_a, l_count * m * n, MPI_INT, 0, Comm);
    MPI_Scatter(mat_b, l_count * n * p, MPI_INT, l_b, l_count * m * n, MPI_INT, 0, Comm);
    MPI_Barrier(Comm);

    // Local matrices for each process
    int m_a[l_count][m][n];
    int m_b[l_count][n][p];
    int m_c[l_count][m][p];

    // Populate local matrices
    int idx = 0;
    for (int c = 0; c < l_count; c++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                m_a[c][i][j] = l_a[idx++];

    idx = 0;
    for (int c = 0; c < l_count; c++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                m_b[c][i][j] = l_b[idx++];

    // Matrix multiplication
    t_start = MPI_Wtime();
    for (int c = 0; c < l_count; c++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                m_c[c][i][j] = 0;
                for (int k = 0; k < p; k++)
                {
                    m_c[c][i][j] += m_a[c][i][k] * m_b[c][k][j];
                }
            }
        }
    }
    t_end = MPI_Wtime();
    printf("PID-%d: Total Time: %f\n", id, t_end - t_start);

    // Gather results
    idx = 0;
    for (int c = 0; c < l_count; c++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                l_c[idx++] = m_c[c][i][j];
            }
        }
    }
    MPI_Gather(l_c, l_count * m * p, MPI_INT, mat_c, l_count * m * p, MPI_INT, 0, Comm);
    MPI_Barrier(Comm);

    if (id == 0)
    {
        // Display results if needed
    }

    MPI_Finalize();
    free(l_a);
    free(l_b);
    free(l_c);
    if (id == 0)
    {
        free(mat_a);
        free(mat_b);
        free(mat_c);
    }
    return 0;
}