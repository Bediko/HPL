#include "second.c"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int matrix_size = 8;
int testing = 1;
void print_matrix(double **matrix)
{
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            printf("%3.f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


void columnpivot(double **matrix, int k)
{
    int s;
    double pivot;
    double *tmp;
    //first element of column=pivot
    pivot = fabs(matrix[k][k]);
    //for all rows > k
    s = k;
    for (; s < matrix_size; s++)
    {
        //if |element| of row > pivot
        if (fabs(matrix[s][k]) > pivot)
        {
            //set new pivot and swap rows
            pivot = fabs(matrix[s][k]);
            tmp = matrix[k];
            matrix[k] = matrix[s];
            matrix[s] = tmp;
        }
    }
};

void test()
{
    double **A;
    double **L;
    double **U;

    //allocate matrix A, L, U
    A = (double **)malloc(matrix_size * sizeof(double *));
    L = (double **)malloc(matrix_size * sizeof(double *));
    U = (double **)malloc(matrix_size * sizeof(double *));
    if (A == NULL || U == NULL || L == NULL)
    {
        printf("Out of Memory\n");
        exit(0);
    }
    for (int i = 0; i < matrix_size; i++)
    {
        A[i] = (double *)malloc(matrix_size * sizeof(double));
        L[i] = (double *)malloc(matrix_size * sizeof(double));
        U[i] = (double *)malloc(matrix_size * sizeof(double));
        if (A[i] == NULL || L[i] == NULL || U[i] == NULL)
        {
            printf("Out of Memory\n");
            exit(0);
        }
    }


    int numprocs, myrank, i;


    // query the number of procs
    int err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    if (err != MPI_SUCCESS)
        printf("error: comm_size (%d)\n", err);


    // query my rank
    err = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (err != MPI_SUCCESS)
        printf("error: comm_rank (%d)\n", err);

    //first process fills Matrix A

    if (myrank == 0)
    {
        for (int i = 0; i < matrix_size; i++)
        {
            A[0][i] = 12;
        }
        for (int i = 1; i < matrix_size; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                if (i == j)
                    A[i][j] = 3;
                else
                    A[i][j] = 12;
            }

        }
        print_matrix(A);

        //LU_decomposition
        for (int k = 0; k < matrix_size; k++)
        {
            //pivotsearch
            columnpivot(A, k);
            for(int i=k;i<matrix_size;i++){
            	L[i][k]=A[i][k]/A[k][k];
            }
            for(int j=k;j<matrix_size;j++){
            	for(int i=k+1;i<matrix_size;i++){
            		A[i][j]=A[i][j]-L[i][k]*A[k][j];
            	}
            }
            print_matrix(L);
        }
        print_matrix(A);
        print_matrix(L);




    }

    MPI_Finalize();
    exit(0);

}

int main(int argc, char **argv)
{
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS)
        printf("error: initializing MPI (%d)\n", err);

    //create NxN matrices
    if (testing)
        test();
}