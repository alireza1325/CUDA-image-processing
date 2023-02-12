

#include <iostream>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <limits.h>
#include <fstream>
#include <cassert>
#include <math.h>
//#include <sys/time.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#include <string>

//#include <opencv2/opencv.hpp>  
#include <opencv2/core/cuda/common.hpp>


void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop)
{
	int num_cols = in.cols;
	int num_rows = in.rows;
	int dummy = 0;
	//out = in.clone();
	std::vector<double> channel;
	channel.resize(3);
	double avg = 0;
	double n_pixel = 0;
	for (int irow = rowstart; irow < rowstop; irow++)
	{
		for (int icol = 0; icol < num_cols; icol++)
		{
			for (int blur_row = irow - level; blur_row < irow + level; blur_row++)
			{
				for (int blur_col = icol - level; blur_col < icol + level; blur_col++)
				{
					if (blur_row >= 0 && blur_row < num_rows && blur_col >= 0 && blur_col < num_cols)
					{
						channel[0] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[0];
						channel[1] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[1];
						channel[2] += (double)in.at<cv::Vec3b>(blur_row, blur_col).val[2];
						n_pixel++; // count the number of pixel values added
					}

				}
			}

			if (n_pixel != 0)
			{
				for (int i = 0; i < channel.size(); i++)
				{
					avg = (double)(channel[i] / n_pixel);
					assert(avg <= 255);
					assert(n_pixel < ((2 * level + 1)* (2 * level + 1)));
					out.at<cv::Vec3b>(irow, icol).val[i] = (uchar)avg;
					channel[i] = 0;
				}
				n_pixel = 0;
			}
		}
	}
}



//---------------------------------------
// CUDA C++ display CUDA device properties
//---------------------------------------
void printCUDADevice(cudaDeviceProp properties)
{
	std::cout << "CUDA Device: " << std::endl;
	std::cout << "\tDevice name              : " << properties.name << std::endl;
	std::cout << "\tMajor revision           : " << properties.major << std::endl;
	std::cout << "\tMinor revision           : " << properties.minor << std::endl;
	std::cout << "\tGlobal memory            : " << properties.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " Gb" << std::endl;
	std::cout << "\tShared memory per block  : " << properties.sharedMemPerBlock / 1024.0 << " Kb" << std::endl;
	std::cout << "\tRegisters per block      : " << properties.regsPerBlock << std::endl;
	std::cout << "\tWarp size                : " << properties.warpSize << std::endl;
	std::cout << "\tMax threads per block    : " << properties.maxThreadsPerBlock << std::endl;
	std::cout << "\tMaximum x dim of block   : " << properties.maxThreadsDim[0] << std::endl;
	std::cout << "\tMaximum y dim of block   : " << properties.maxThreadsDim[1] << std::endl;
	std::cout << "\tMaximum z dim of block   : " << properties.maxThreadsDim[2] << std::endl;
	std::cout << "\tMaximum x dim of grid    : " << properties.maxGridSize[0] << std::endl;
	std::cout << "\tMaximum y dim of grid    : " << properties.maxGridSize[1] << std::endl;
	std::cout << "\tMaximum z dim of grid    : " << properties.maxGridSize[2] << std::endl;
	std::cout << "\tClock frequency          : " << properties.clockRate / 1000.0 << " MHz" << std::endl;
	std::cout << "\tConstant memory          : " << properties.totalConstMem << std::endl;
	std::cout << "\tNumber of multiprocs     : " << properties.multiProcessorCount << std::endl;
}





/**
 * @brief      The blure kernel
 *
 *
 * @param      input    The input image
 * @param      output   The output image
 * @param[in]  width    The image width
 * @param[in]  height   The image height
 * @param[in]  inStep   The input step
 * @param[in]  blur level
 */
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int inStep,int level) {
	// Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Number of channels
	const int in_c = inStep / width;
	
	// Only valid threads perform memory I/O
	if ((x < width) && (y < height)) {

		// Location of pixel
		int in_loc = y * inStep + (in_c * x);
		int out_loc = in_loc;

		double channel[3];
		int n_pixel = 0;

		for (int blur_row = y - level; blur_row < y + level; blur_row++)
		{
			for (int blur_col = x - level; blur_col < x + level; blur_col++)
			{
				if (blur_row >= 0 && blur_row < height && blur_col >= 0 && blur_col < width)
				{
					in_loc = blur_row * inStep + (in_c * blur_col);
					for (int i = 0; i < in_c; ++i) {
						channel[i] += (double)(input[in_loc + i]);
					}
					n_pixel++; // count the number of pixel values added
				}
			}
		}
		double avg = 0;
		if (n_pixel != 0)
		{
			for (int i = 0; i < in_c; i++)
			{
				avg = (double)(channel[i] / n_pixel);
				assert(avg <= 255);
				assert(n_pixel < ((2 * level + 1)*(2 * level + 1)));
				output[out_loc + i] = avg;
				channel[i] = 0;
			}
			n_pixel = 0;
		}
	}
}



int main(int argc, char** argv) 
{

	// Collect inputs
	if (argc < 5)
	{
		std::cerr << "Required Comamnd-Line Arguments Are:\n";
		std::cerr << "Image file name\n";
		std::cerr << "Level of blur\n";
		std::cerr << "Block size in x and y dim\n";
		return -1;
	}
	
	char* imagePath = argv[1];
	int level = atoi(argv[2]);
	int blockx = atoi(argv[3]);
	int blocky = atoi(argv[4]);

	// device information
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, 0);
	printCUDADevice(device_properties);

	cv::Mat input = cv::imread(imagePath, 1);

	if (input.empty()) {
		std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}
	// print image details 
	std::cout << "Image details are: "<< std::endl;
	std::cout << "input.rows " << input.rows << std::endl;
	std::cout << "input.cols " << input.cols << std::endl;
	std::cout << "Input.step " << input.step << std::endl;

	// Create output image
	cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

	// Calculate total number of bytes of input and output image
	const int inBytes = input.step * input.rows;
	const int outBytes = output.step * output.rows;

	unsigned char* b_input, * b_output;

	b_input = (unsigned char*)malloc(inBytes);
	b_output = (unsigned char*)malloc(outBytes);

	// Start timings
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Allocate device memory
	cudaSafeCall(cudaMalloc<unsigned char>(&b_input, inBytes));
	cudaSafeCall(cudaMalloc<unsigned char>(&b_output, outBytes));

	// Copy data from OpenCV input image to device memory
	cudaSafeCall(cudaMemcpy(b_input, input.ptr(), inBytes, cudaMemcpyHostToDevice));

	// Specify a reasonable block size
	const dim3 block_size(blockx, blocky);
	
	// Calculate grid size to cover the whole image
	int gridy = (int)ceil((double)input.rows / (double)block_size.y);
	int gridx = (int)ceil((double)input.cols / (double)block_size.x);
	dim3 grid_size(gridx, gridy);

	// Launch kernel
	blur_kernel << <grid_size, block_size >> > (b_input, b_output, input.cols, input.rows, input.step,level);

	// Synchronize to check for any kernel launch errors
	cudaSafeCall(cudaDeviceSynchronize());

	// Copy back data from destination device meory to OpenCV output image
	cudaSafeCall(cudaMemcpy(output.ptr(), b_output, outBytes, cudaMemcpyDeviceToHost));

	// Free the memory
	cudaSafeCall(cudaFree(b_input));
	cudaSafeCall(cudaFree(b_output));

	// Stop timings
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_time;
	cudaEventElapsedTime(&gpu_time, start, stop);   //time in milliseconds

	gpu_time /= 1000.0;

	// Destroy time objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// cpu processing and timings
	double cpu_t_start = (double)clock() / (double)CLOCKS_PER_SEC;
	cv::Mat cpublurredimg = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
	
	imageBlur(input, cpublurredimg, level, 0, input.rows);
	double cpu_time = (double)clock() / (double)CLOCKS_PER_SEC - cpu_t_start;
	
	// Error calculations
	double error = 0;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			error += fabs((double)output.at<cv::Vec3b>(i, j).val[0] - (double)cpublurredimg.at<cv::Vec3b>(i, j).val[0]);
			error += fabs((double)output.at<cv::Vec3b>(i, j).val[1] - (double)cpublurredimg.at<cv::Vec3b>(i, j).val[1]);
			error += fabs((double)output.at<cv::Vec3b>(i, j).val[2] - (double)cpublurredimg.at<cv::Vec3b>(i, j).val[2]);
		}
	}


	// Print the results
	std::cout << "GPU Time = " << gpu_time << std::endl;
	std::cout << "CPU Time = " << cpu_time << std::endl;
	std::cout << "Speedup = " << cpu_time / gpu_time << std::endl;
	std::cout << "Error = " << error << std::endl;

	// Show the input and outputs
	cv::imshow("Input image", input);
	cv::imshow("Blurred image on GPU", output);
	cv::imshow("Blurred image on CPU", cpublurredimg);
	// Wait for key press
	cv::waitKey();

	return 0;
}
