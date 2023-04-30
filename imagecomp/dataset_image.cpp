#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    int num_processes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


     // Load the dataset
    string filename = "data_batch_4.bin";
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "Error: could not open file " << filename << endl;
        return 1;
    }

    // Read the dataset into a vector of Mat objects
    vector<Mat> images;
    vector<int> labels;
    int num_images = 0;
    while (!file.eof()) {
        // Read the label
        char label_char;
        file.read(&label_char, 1);
        int label = label_char;

        // Read the image
        Mat image(32, 32, CV_8UC3);
        file.read((char*)image.data, 32 * 32 * 3);
        images.push_back(image);
        labels.push_back(label);
        num_images++;
    }
    file.close();

    // Split the dataset among the processes
    int num_images_per_process = num_images / num_processes;
    int start_index = rank * num_images_per_process;
    int end_index = (rank == num_processes - 1) ? num_images - 1 : start_index + num_images_per_process - 1;
    int num_images_this_process = end_index - start_index + 1;

    // Create an array to store the distances between the test image and each training image
    double* distances = new double[num_images_this_process];

   // Load the test image
Mat test_image = imread("cattest.jpeg");

// Check if the image was loaded successfully
if (test_image.empty()) {
    cout << "Error: could not load test image" << endl;
    return 1;
}

// Resize the image to match the CIFAR-10 image size (32x32)
resize(test_image, test_image, Size(32, 32));

// Convert the image to grayscale
cvtColor(test_image, test_image, COLOR_BGR2GRAY);

// Convert the image to the same format as the CIFAR-10 images (32x32x3, CV_8UC3)
cvtColor(test_image, test_image, COLOR_GRAY2BGR);


    // Calculate the distance between the test image and each training image
    for (int i = start_index; i <= end_index; i++) {
        double distance = 0.0;
        for (int j = 0; j < 32 * 32 * 3; j++) {
            double diff = images[i].data[j] - test_image.data[j];
            distance += diff * diff;
        }
        distances[i - start_index] = sqrt(distance);
    }

    // Gather the distances from all processes to the root process
    double* all_distances = new double[num_images];
    MPI_Gather(distances, num_images_this_process, MPI_DOUBLE, all_distances, num_images_this_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Find the index of the image with the minimum distance
    int min_index;
    double min_distance = numeric_limits<double>::max();
    if (rank == 0) {
        for (int i = 0; i < num_images; i++) {
            if (all_distances[i] < min_distance) {
                min_distance = all_distances[i];
                min_index = i;
            }
        }
    }

    // Broadcast the index of the image with the minimum distance to all processes
    MPI_Bcast(&min_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

 // Calculate the total distance between the test image and all training images
double sum_distance = 0.0;
for (int i = 0; i < num_images; i++) {
    sum_distance += all_distances[i];
}

// Calculate the confidence of the result
double confidence = 0.0;
if (sum_distance != 0.0) {
    // Calculate the percentage of the total distance that is due to the closest match
    double percentage_distance_values = min_distance / sum_distance;
    // Calculate the confidence as the proportion of the dataset for which the closest match is the test image
    confidence = 1.0 - (percentage_distance_values * num_images);
    // Ensure that the confidence is not negative
    if (confidence < 0) {
        confidence = 0.0;
    }
} else {
    // If the sum of distances is zero, the test image is identical to the training image with the closest match
    confidence = 1.0;
}

// Print the confidence of the result
cout << "Confidence: " << confidence << endl;


// Clean up
delete[] distances;
delete[] all_distances;
MPI_Finalize();

return 0;
}

