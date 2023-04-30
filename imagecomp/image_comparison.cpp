#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <thread>


using namespace cv;
using namespace std;

// Algorithm 1: Calculate difference between two pixels x and y
int diff(int x, int y) {
    return pow(x - y, 2);
}

// Algorithm 2: Calculate sum of differences between embedded pixels of two 1D arrays
int sumOfDiffs1D(int *x, int *y, int N) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += diff(x[i], y[i]); // add the difference between pixels to sum
    }
    return sum;
}

// Algorithm 3: Calculate sum of differences between embedded pixels of two 2D arrays
int sumOfDiffs2D(int **x, int **y, int M, int N) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int j = 0; j < M; j++) { // loop through rows of arrays
        sum += sumOfDiffs1D(x[j], y[j], N); // add the sum of differences in each row to the total sum
    }
    return sum;
}

// Algorithm 4: Read two images and convert them to 2D grayscale arrays
void readImages(cv::Mat& img1, cv::Mat& img2, int**& arr1, int**& arr2, int& M, int& N) {
    cv::Mat img1_gray, img2_gray;
    img1 = cv::imread("apple.jpeg", cv::IMREAD_GRAYSCALE); // read first image as grayscale
    img2 = cv::imread("apple2.jpeg", cv::IMREAD_GRAYSCALE); // read second image as grayscale
    M = img1.rows; // set the number of rows to M
    N = img1.cols; // set the number of columns to N
    arr1 = new int*[M]; // dynamically allocate memory for first array
    arr2 = new int*[M]; // dynamically allocate memory for second array
    for (int i = 0; i < M; i++) { // loop through rows
        arr1[i] = new int[N]; // allocate memory for each row in first array
        arr2[i] = new int[N]; // allocate memory for each row in second array
        for (int j = 0; j < N; j++) { // loop through columns
            arr1[i][j] = img1.at<uchar>(i,j); // convert pixel value to grayscale value and store in first array
            arr2[i][j] = img2.at<uchar>(i,j); // convert pixel value to grayscale value and store in second array
        }
    }
}

// Algorithm 5: Calculate percentage distance value of two 2D arrays
float percentDist(int **arr1, int **arr2, int M, int N) {
    double Ed2 = sumOfDiffs2D(arr1, arr2, M, N); // using double to avoid integer overflow
    double percentDist = (Ed2 / (M * N)) / 255.0; //
 percentDist = percentDist * 100; // multiplying by 100 to get the percentage value
    if (percentDist < 0) { // check if percentage is negative
        percentDist = 0; // set it to 0, since percentage cannot be negative
    }
    return percentDist;
}
// Function to free dynamically allocated 2D arrays
void free2DArray(int **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] arr[i];
    }
    delete[] arr;
}


int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get rank and size of the current process in the MPI_COMM_WORLD communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Declare variables to hold the two input images and their pixel values
    Mat img1, img2;
    int **arr1, **arr2, M, N;

    // Read the input images and their pixel values into memory
    readImages(img1, img2, arr1, arr2, M, N);

    // Create an array of threads, one for each process in the MPI communicator
    std::thread threads[size];

    // Calculate the local sum of differences between the pixel values of the two images using threads
    // Each thread calculates the sum of differences for its own subregion of the image
    int chunk_size = M / size;
    for (int i = 0; i < size; i++) {
        threads[i] = std::thread([&arr1, &arr2, chunk_size, N](int rank) {
            int start_row = rank * chunk_size;
            int end_row = (rank + 1) * chunk_size;
            int localSum = sumOfDiffs2D(arr1 + start_row, arr2 + start_row, end_row - start_row, N);
            MPI_Send(&localSum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }, i);
    }

    // Wait for all threads to finish
    for (int i = 0; i < size; i++) {
        threads[i].join();
    }

    // Reduce all local sums to the global sum using MPI_Reduce
    int globalSum = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int localSum;
            MPI_Recv(&localSum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            globalSum += localSum;
        }
    }

    // Calculate the percentage distance value between the two images and print it if the current process is the root process (rank 0)
    if (rank == 0) {
        float percent = percentDist(arr1, arr2, M, N);
        cout << "Percentage Distance Value: " << percent << "%" << endl;

        // Check if the images are similar or not based on the threshold value
        float threshold = 10.0;
        if (percent <= threshold) {
            cout << "The images are similar." << endl;
        } else {
            cout << "The images are not similar." << endl;
        }
    }

    // Free dynamically allocated memory for the pixel values of the two images
    free2DArray(arr1, M);
    free2DArray(arr2, M);

// Finalize MPI environment
MPI_Finalize();

return 0;
}