#include "second.c"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int matrix_size = 4;
int testing = 1;
void print_matrix(double matrix[][matrix_size])
{
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            printf("%3.3lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vec(double vec[matrix_size])
{
    for (int i = 0; i < matrix_size; i++)
        printf("%3.3lf \n", vec[i]);
    printf("\n");
}


void columnpivot(double matrix[][matrix_size], int k, double vec[matrix_size])
{
    int s, p;
    double pivot;
    double tmp, tmp2;
    //first element of column=pivot
    pivot = fabs(matrix[k][k]);
    //for all rows > k
    s = k;
    p = k;
    for (; s < matrix_size; s++)
    {
        //if |element| of row > pivot
        if (fabs(matrix[s][k]) > pivot)
        {
            pivot = fabs(matrix[s][k]);
            p = s;
        }
    }
    if (p != k)
    {
        //swap matrix rows
        for(int i=0;i<matrix_size;i++){
        tmp = matrix[k][i];
        matrix[k][i] = matrix[p][i];
        matrix[p][i] = tmp;
    }
        //swap vector elemnts
        tmp2 = vec[k];
        vec[k] = vec[p];
        vec[p] = tmp2;
    }
}

void single()
{
    
    double vec[matrix_size],veco[matrix_size];



    //allocate matrix A, L, U
    double A[matrix_size][matrix_size], O[matrix_size][matrix_size];



    //first process fills Matrix A

    // for (int i = 0; i < matrix_size; i++)
    // {
    //     A[0][i] = 12;
    // }
    // for (int i = 1; i < matrix_size; i++)
    // {
    //     for (int j = 0; j < matrix_size; j++)
    //     {
    //         if (i == j)
    //             A[i][j] = 3;
    //         else
    //             A[i][j] = 12;
    //     }

    // }

    //Testcase: step by step solution http://www.das-gelbe-rechenbuch.de/download/Lu.pdf page 8 ff
    A[0][0] = 6.0;
    A[0][1] = 5.0;
    A[0][2] = 3.0;
    A[0][3] = -10.0;
    A[1][0] = 3.0;
    A[1][1] = 7.0;
    A[1][2] = -3.0;
    A[1][3] = 5.0;
    A[2][0] = 12.0;
    A[2][1] = 4.0;
    A[2][2] = 4.0;
    A[2][3] = 4.0;
    A[3][0] = 0.0;
    A[3][1] = 12.0;
    A[3][2] = 0.0;
    A[3][3] = -8.0;
    vec[0] = -10;
    vec[1] = 14;
    vec[2] = 8;
    vec[3] = -8;

    //copy A and b to use original values as test
    for(int i=0; i<matrix_size;i++){
    	for(int j=0;j<matrix_size;j++){
    		O[i][j]=A[i][j];
    	}
    	veco[i]=vec[i];
    }

    printf("Input\nb:\n");
    print_vec(vec);
    printf("A:\n");
    print_matrix(A);
    

    //LU_decomposition in Matrix A
    for (int i = 0; i < matrix_size - 1; i++)
    {

        //pivotsearch
        columnpivot(A, i, vec);
        for (int k = i + 1; k < matrix_size; k++)
        {   //L
            A[k][i] = A[k][i] / A[i][i];
            


            for (int j = i + 1; j < matrix_size; j++)
            {
                //U
                A[k][j] = A[k][j] - (A[k][i] * A[i][j]);

            }
            //Ly=z
            vec[k]=vec[k]-(A[k][i]*vec[i]);
        }

    }

    print_matrix(A);
    print_vec(vec);
    
    

    //Ux=y
    for (int i = matrix_size-1; i >=0; i--)
    {
        double sum = 0;
        for (int j = matrix_size-1; j > i; j--)
        {
            
                sum += A[i][j] * vec[j];
        }
        vec[i] = (vec[i]-sum)/A[i][i];
        
    }
    printf("X\n");
    print_vec(vec);
    printf("Ax:\n");
    for(int i=0; i<matrix_size;i++){
    	double sum=0;
    	for(int j=0;j<matrix_size;j++){
    		sum+=O[i][j]*vec[j];
    	}
    	printf("%3.3lf\n",sum);
    }
}
void test()
{
    int numprocs, myrank, i;
    int err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    if (err != MPI_SUCCESS)
        printf("error: comm_size (%d)\n", err);


    // query my rank
    err = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (err != MPI_SUCCESS)
        printf("error: comm_rank (%d)\n", err);
    if (myrank == 0)
    {
        single();
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