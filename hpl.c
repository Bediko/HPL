/**
 * Implementation of HPL-Benchmark
 */

#include "second.c"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>



int matrix_size = 4;
int testing = 1;
int myrank;

/**
 * @brief prints a N x N matrix
 *
 * @param matrix [in] N x N matrix
 */
void print_matrix(double matrix[][matrix_size])
{
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            printf("% 3.3e ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief prints vector of size N
 *
 *
 * @param vec [in] vector of size N
 */
void print_vec(double vec[matrix_size])
{
    for (int i = 0; i < matrix_size; i++)
        printf("% lf \n", vec[i]);
    printf("\n");
}


/**
 * @brief calculates the pivot of a column and swaps the rows.
 *
 *
 * @param matrix [in] N x N matrix
 * @param k [in] column to check
 * @param vec [in] Vector of size N
 */
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
        for (int i = 0; i < matrix_size; i++)
        {
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
void columnpivot_parallel(double matrix[][matrix_size], int k, double vec[matrix_size])
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

        MPI_Bcast( &p, 1, MPI_INT, myrank, MPI_COMM_WORLD);
        //swap matrix rows
        for (int i = 0; i < matrix_size; i++)
        {
            tmp = matrix[k][i];
            matrix[k][i] = matrix[p][i];
            matrix[p][i] = tmp;
        }
        //swap vector elemnts
        tmp2 = vec[k];
        vec[k] = vec[p];
        vec[p] = tmp2;
    }
    else
    {
        p = -1;
        MPI_Bcast( &p, 1, MPI_INT, myrank, MPI_COMM_WORLD);
    }
}

/**
 * @brief test on single computer
 * @details Test function fpr LU decomposition on one computer. Calculates the LU-decomposition in place and
 * checks if Ax=b for the calculated x.
 */
void single()
{
    if (myrank != 0)
        return;

    double vec[matrix_size], veco[matrix_size];



    //allocate matrix A, L, U
    double A[matrix_size][matrix_size], O[matrix_size][matrix_size];




    // srand(time(NULL));
    // //testcase: random values
    // for (int i = 0; i < matrix_size; i++)
    // {
    //     for (int j = 0; j < matrix_size; j++)
    //     {
    //         A[i][j] = rand() % 100;
    //     }
    //     vec[i] = rand() % 100;

    // }

    //Testcase: step by step solution http://www.das-gelbe-rechenbuch.de/download/Lu.pdf page 8 ff
    // A[0][0] = 6.0;
    // A[0][1] = 5.0;
    // A[0][2] = 3.0;
    // A[0][3] = -10.0;
    // A[1][0] = 3.0;
    // A[1][1] = 7.0;
    // A[1][2] = -3.0;
    // A[1][3] = 5.0;
    // A[2][0] = 12.0;
    // A[2][1] = 4.0;
    // A[2][2] = 4.0;
    // A[2][3] = 4.0;
    // A[3][0] = 0.0;
    // A[3][1] = 12.0;
    // A[3][2] = 0.0;
    // A[3][3] = -8.0;
    // vec[0] = -10;
    // vec[1] = 14;
    // vec[2] = 8;
    // vec[3] = -8;

    //copy A and b to use original values as test
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            O[i][j] = A[i][j];
        }
        veco[i] = vec[i];
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
        {
            //L
            A[k][i] = A[k][i] / A[i][i];



            for (int j = i + 1; j < matrix_size; j++)
            {
                //U
                A[k][j] = A[k][j] - (A[k][i] * A[i][j]);

            }
            //Ly=z
            vec[k] = vec[k] - (A[k][i] * vec[i]);
        }

    }
    printf("Z\n");
    print_vec(vec);




    //Ux=y
    for (int i = matrix_size - 1; i >= 0; i--)
    {
        double sum = 0;
        for (int j = matrix_size - 1; j > i; j--)
        {

            sum += A[i][j] * vec[j];
        }
        vec[i] = (vec[i] - sum) / A[i][i];

    }
    printf("X\n");
    print_vec(vec);
    printf("Ax:\n");
    double erg[matrix_size];
    int succ = 1;
    //Ax=b
    for (int i = 0; i < matrix_size; i++)
    {
        erg[i] = 0;
        for (int j = 0; j < matrix_size; j++)
        {
            erg[i] += O[i][j] * vec[j];
        }
        //big vectors are work to compare by hand, use simple way to compare automated, not always right.
        if ( (fabs(erg[i]) - fabs(veco[i])) > 0.0000000001)
        {
            printf("Wrong at index %i: % lf\t%lf\n", i, erg[i], veco[i]);
            succ = 0;
        }
    }
    if (succ)
        printf("check succesfull, Ax=b is true\n");


}

/**
 * @brief transposes a N x N matrix
 * @param matrix [in]  N x N matrix to transpose
 */
void transpose_matrix(double matrix[][matrix_size])
{
    double temp;
    for (int i = 0; i < matrix_size - 1; i++)
    {
        for (int k = i + 1; k < matrix_size; k++)
        {
            temp = matrix[i][k];
            matrix[i][k] = matrix[k][i];
            matrix[k][i] = temp;
        }
    }
}
/**
 * @brief Swaps row a wit row b in matrix
 *
 * @param matrix [in] matrix containing rows
 * @param a [in] first row to swap
 * @param b [in] second row to swap
 */
void swap_rows(double matrix[][matrix_size], int a, int b)
{
    double temp;
    temp = matrix[a][b];
    matrix[a][b] = matrix[b][a];
    matrix[b][a] = temp;

}


/**
 * @brief calculates Ax=b with LU-decomposition in parallel
 */
void parallel(int argc, char **argv)
{


    // int err = MPI_Init(&argc, &argv);
    // if (err != MPI_SUCCESS)
    //     printf("error: initializing MPI (%d)\n", err);


    int numprocs;
    int i, j;
    int err;
    MPI_Status status;

    // query number of procs
    err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    if (err != MPI_SUCCESS)
        printf("error: comm_size (%d)\n", err);

    // query my rank
    // err = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // if (err != MPI_SUCCESS)
    //     printf("error: comm_rank (%d)\n", err);

    //every process gets a whole matrix and vector
    double vec[matrix_size], veco[matrix_size];
    int columns = matrix_size / numprocs;
    double A[matrix_size][matrix_size], O[matrix_size][matrix_size];

    MPI_Datatype temp, columnsToSend;


    vec[0] = -10;
    vec[1] = 14;
    vec[2] = 8;
    vec[3] = -8;

    if (myrank == 0)
    {
        printf("Number of procs: %i\n", numprocs);

        A[0][0] = 6.0;
        A[1][0] = 3.0;
        A[2][0] = 12.0;
        A[3][0] = 0.0;

    }
    if (myrank == 1)
    {
        A[0][1] = 5.0;
        A[1][1] = 7.0;
        A[2][1] = 4.0;
        A[3][1] = 12.0;

    }
    if (myrank == 2)
    {

        A[0][2] = 3.0;
        A[1][2] = -3.0;
        A[2][2] = 4.0;
        A[3][2] = 0.0;

    }
    if (myrank == 3)
    {
        A[0][3] = -10.0;
        A[1][3] = 5.0;
        A[2][3] = 4.0;
        A[3][3] = -8.0;

    }
    j = 0;

    for(int i=0;i<matrix_size;i++){
        for(int j=0;j<matrix_size;j++){
            O[i][j]=A[i][j];
        }
        veco[i]=vec[i];
    }

    int pivot;
    double tmp;
    for (int i = 0; i < matrix_size - 1; i++)
    {
        //begin pivotsearch
        if (i % numprocs == myrank)
        {
            columnpivot_parallel(A, i, vec);
            j++;
        }
        else
        {
            MPI_Bcast( &pivot, 1, MPI_INT, i % numprocs, MPI_COMM_WORLD);

            if (pivot > -1)
            {
                for (int l = 0; l < matrix_size; l++)
                {
                    tmp = A[i][l];
                    A[i][l] = A[pivot][l];
                    A[pivot][l] = tmp;

                }
                tmp=vec[i];
                vec[i]=vec[pivot];
                vec[pivot]=tmp;
            }

        }
        //end pivot



        //LU
        if (i % numprocs == myrank)
        {
            double elements[matrix_size - (i + 1)];
            for (int k = i + 1; k < matrix_size; k++)
            {
                //L
                A[k][i] = A[k][i] / A[i][i];
                elements[k - (i + 1)] = A[k][i];
            }

            MPI_Bcast(elements, matrix_size - (i + 1), MPI_DOUBLE, myrank, MPI_COMM_WORLD);
        }
        else
        {
            double elements[matrix_size - (i + 1)];
            MPI_Bcast(elements, matrix_size - (i + 1), MPI_DOUBLE, i % numprocs, MPI_COMM_WORLD);
            for (int k = 0; k < matrix_size - (i + 1); k++)
            {
                //L
                A[k + i + 1][i] = elements[k];
            }
        }

        for (int j = i + 1; j < matrix_size; j++)
        {
            if (j % numprocs == myrank)
            {
                for (int k = i + 1; k < matrix_size; k++)
                    A[k][j] = A[k][j] - (A[k][i] * A[i][j]);
            }
            //U
             vec[j] = vec[j] - (A[j][i] * vec[i]);
        }

    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // sleep(myrank);
    // printf("Rank: %d\n", myrank);
    // print_vec(vec);
    

    double lsum, gsum;
    //Ux=y
    for (int i = matrix_size-1; i >= 0; i--)//row
    {
        //forward subtitution method
        lsum = 0.0; //local sum
        gsum = 0.0; //global sum
        //run over the upper triangle
        for (int p = matrix_size-1; p > i; p--)//column
        {
             if (p % numprocs == myrank) //If the column is in my process
                 lsum += A[i][p] * vec[p]; //add to my local sum
        }
        MPI_Allreduce(&lsum, &gsum, 1,  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(i %numprocs==myrank){
        	vec[i] = (vec[i] - gsum)/A[i][i]; 
        }
        else
        	vec[i]=0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(myrank);
    printf("Rank X : %d\n", myrank);
    print_vec(vec);

    //Ax=b
    double erg[matrix_size],sum;
    int succ=1;
    for (int i = 0; i < matrix_size; i++)//row
    {
        sum = 0;
        for (int j = 0; j < matrix_size; j++)//column
        {
            if (j % numprocs == myrank)
                sum += O[i][j] * vec[j];
        }
        MPI_Allreduce(&sum, &sum, 1,  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(myrank==0)
            erg[i]=sum;
        //big vectors are work to compare by hand, use simple way to compare automated, not always right.
        if ( (fabs(erg[i]) - fabs(veco[i])) > 0.0000000001)
        {
            printf("Wrong at index %i: % lf\t%lf\n", i, erg[i], veco[i]);
            succ=0;
        }
    }
    if(myrank==0&&succ)
        printf("Check Succesfull, Ax=b is true");


    MPI_Finalize();
    exit(0);
}

int main(int argc, char **argv)
{
    if (argc > 1)
        matrix_size = atoi(argv[1]);



    //create NxN matrices
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS)
        printf("error: initializing MPI (%d)\n", err);
    err = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (err != MPI_SUCCESS)
        printf("error: comm_rank (%d)\n", err);


    if (testing)
    {
        single();
        MPI_Barrier(MPI_COMM_WORLD);
        parallel(argc, argv);
    }
    MPI_Finalize();
}
