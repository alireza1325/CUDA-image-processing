# CUDA-image-processing-
In this repo, the image blur function is implemented using CUDA and OpenCV. 

In this lab, the image blur function that was previously developed for CPU processing and MPI is converted for CUDA processing. The CUDA and CPU processing time for blurring an image and the speedup were computed. Moreover, blurred images calculated by CPU and GPU are illustrated in separate image windows for comparison. In the following sections, the details of the code will be revealed, and answers to the lab questions will be provided.

## Execution configuration (thread layout) for CUDA kernel

The thread layout was configured as 2-D because our problem has two dimensions, and we need to access rows and columns of an image. In other words, to move across image we need two separate indexes for rows and columns. To set up a 2-D layout configuration for threads within blocks, first, the block size was read through command line arguments, and we can specify the block size by the following command:
const dim3 block_size(blockx, blocky);

Then, to make sure we have enough indexes to cover the whole image, the grid size over x and y dimensions (the number of blocks) was specified by the following lines of code:
int gridy = (int)ceil((double)input.rows / (double)block_size.y);
int gridx = (int)ceil((double)input.cols / (double)block_size.x);
dim3 grid_size(gridx, gridy);

This is the output of the code by 16*16 block size and blur level 3:

![image](https://user-images.githubusercontent.com/57262710/218327140-9b71e675-e537-4952-9fc4-af005abb0929.png)

Here is the GPU and speedup information:

![image](https://user-images.githubusercontent.com/57262710/218327214-b7d81a51-7dbf-4f02-ba46-188c78ce2bd5.png)

As can be seen, the blurred images in the CPU and GPU are the same, and a speedup of around 12 was achieved in this run.

## Computational complexity and speedup 

The code was compiled and run for different block sizes and blur levels, and the results are tabulated in two tables below. As can be seen, the performance of the parallel code significantly depends on the block size. When the block size is 1×1 the GPU performance is very low. It is because of the fact that the blocks are dispatched in wraps, and with one thread per block, we aren’t using SIMD instruction within the streaming multiprocessing. By increasing the block size (decreasing the number of blocks), the speedup is increased up to a block size of 8. From block sizes 8 to 16, the speedup remained relatively the same. However, growing the block size to 32×32 resulted in a minor performance reduction with respect to block sizes 8 and 16. 
At a block size of 8, for example, we have 64 threads in every block (two dimensions) or two warps. In this case, we somehow benefit from coalesced memory access. It seems that for every problem size, there is an optimized block size.

|     Blur level    |     Block size    |     GPU time (s)    |     CPU time (s)    |     speedup    |
|-------------------|-------------------|---------------------|---------------------|----------------|
|     3             |     1×1           |     1.78            |     0.769           |     0.432      |
|     3             |     2×2           |     0.443           |     0.769           |     1.736      |
|     3             |     4×4           |     0.121           |     0.769           |     6.355      |
|     3             |     8×8           |     0.063           |     0.769           |     12.206     |
|     3             |     16×16         |     0.062           |     0.769           |     12.403     |
|     3             |     32×32         |     0.068           |     0.769           |     11.309     |



|     Blur level    |     Block size    |     GPU time (s)    |     CPU time (s)    |     speedup    |
|-------------------|-------------------|---------------------|---------------------|----------------|
|     6             |     1×1           |     6.586           |     2.978           |     0.452      |
|     6             |     2×2           |     1.651           |     2.978           |     1.804      |
|     6             |     4×4           |     0.414           |     2.978           |     7.193      |
|     6             |     8×8           |     0.210           |     2.978           |     14.181     |
|     6             |     16×16         |     0.209           |     2.978           |     14.249     |
|     6             |     32×32         |     0.238           |     2.978           |     12.513     |

Overall, when the blur level is 6, the speedup is higher than compared to the blur level 3. It might be because of increasing the computation with respect to the communication between host and device (same as MPI). However, using a big blur level results in increasing halo pixels and overlapping of memory access.
