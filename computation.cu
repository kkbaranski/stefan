/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                        ( computation.cu )                          //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "computation.h"

__host__ void compute_first() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_INPUT;
	double* d_FIRST_LAYER_WEIGHT;
	double* d_FIRST_LAYER;
	
	cudaMalloc(&d_INPUT, sizeof(INPUT));
	cudaMalloc(&d_FIRST_LAYER_WEIGHT, sizeof(FIRST_LAYER_WEIGHT));
	cudaMalloc(&d_FIRST_LAYER, sizeof(FIRST_LAYER));

	cudaMemcpy(d_INPUT, INPUT, sizeof(INPUT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FIRST_LAYER_WEIGHT, FIRST_LAYER_WEIGHT, sizeof(FIRST_LAYER_WEIGHT), cudaMemcpyHostToDevice);

	int numBlocks = 6;
	dim3 threadsPerBlock(13, 13);
	d_compute_first<<<numBlocks, threadsPerBlock>>>((double (*)[29])d_INPUT, (double (*)[13][13][5][5])d_FIRST_LAYER_WEIGHT, (double (*)[13][13])d_FIRST_LAYER);
	cudaDeviceSynchronize();

	cudaMemcpy(FIRST_LAYER, d_FIRST_LAYER, sizeof(FIRST_LAYER), cudaMemcpyDeviceToHost);

	cudaFree(d_INPUT);
	cudaFree(d_FIRST_LAYER_WEIGHT);
	cudaFree(d_FIRST_LAYER);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "First layer: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}

__host__ void compute_second() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_FIRST_LAYER;
	double* d_SECOND_LAYER_WEIGHT;
	double* d_SECOND_LAYER;
	
	
	cudaMalloc(&d_FIRST_LAYER, sizeof(FIRST_LAYER));
	cudaMalloc(&d_SECOND_LAYER_WEIGHT, sizeof(SECOND_LAYER_WEIGHT));
	cudaMalloc(&d_SECOND_LAYER, sizeof(SECOND_LAYER));

	cudaMemcpy(d_FIRST_LAYER, FIRST_LAYER, sizeof(FIRST_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SECOND_LAYER_WEIGHT, SECOND_LAYER_WEIGHT, sizeof(SECOND_LAYER_WEIGHT), cudaMemcpyHostToDevice);


	int numBlocks = 50;
	dim3 threadsPerBlock(5, 5);
	d_compute_second<<<numBlocks, threadsPerBlock>>>((double (*)[13][13])d_FIRST_LAYER, (double (*)[5][5][6][5][5])d_SECOND_LAYER_WEIGHT, (double (*)[5][5])d_SECOND_LAYER);
	cudaDeviceSynchronize();

	cudaMemcpy(SECOND_LAYER, d_SECOND_LAYER, sizeof(SECOND_LAYER), cudaMemcpyDeviceToHost);

	cudaFree(d_FIRST_LAYER);
	cudaFree(d_SECOND_LAYER_WEIGHT);
	cudaFree(d_SECOND_LAYER);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Second layer: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}

__host__ void compute_third() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_SECOND_LAYER;
	double* d_THIRD_LAYER_WEIGHT;
	double* d_THIRD_LAYER;
	
	cudaMalloc(&d_SECOND_LAYER, sizeof(SECOND_LAYER));
	cudaMalloc(&d_THIRD_LAYER_WEIGHT, sizeof(THIRD_LAYER_WEIGHT));
	cudaMalloc(&d_THIRD_LAYER, sizeof(THIRD_LAYER));

	cudaMemcpy(d_SECOND_LAYER, SECOND_LAYER, sizeof(SECOND_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_THIRD_LAYER_WEIGHT, THIRD_LAYER_WEIGHT, sizeof(THIRD_LAYER_WEIGHT), cudaMemcpyHostToDevice);

	int numBlocks = 100;
	int threadsPerBlock = 50;
	d_compute_third<<<numBlocks, threadsPerBlock>>>((double (*)[5][5])d_SECOND_LAYER, (double (*)[50][5][5])d_THIRD_LAYER_WEIGHT, (double *)d_THIRD_LAYER);
	cudaDeviceSynchronize();

	cudaMemcpy(THIRD_LAYER, d_THIRD_LAYER, sizeof(THIRD_LAYER), cudaMemcpyDeviceToHost);

	cudaFree(d_SECOND_LAYER);
	cudaFree(d_THIRD_LAYER_WEIGHT);
	cudaFree(d_THIRD_LAYER);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Third layer: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}

__host__ void compute_output() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_THIRD_LAYER;
	double* d_OUTPUT_WEIGHT;
	double* d_OUTPUT;
	
	cudaMalloc(&d_THIRD_LAYER, sizeof(THIRD_LAYER));
	cudaMalloc(&d_OUTPUT_WEIGHT, sizeof(OUTPUT_WEIGHT));
	cudaMalloc(&d_OUTPUT, sizeof(OUTPUT));

	cudaMemcpy(d_THIRD_LAYER, THIRD_LAYER, sizeof(THIRD_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OUTPUT_WEIGHT, OUTPUT_WEIGHT, sizeof(OUTPUT_WEIGHT), cudaMemcpyHostToDevice);

	int numBlocks = 1;
	int threadsPerBlock = 10;
	d_compute_output<<<numBlocks, threadsPerBlock>>>((double *)d_THIRD_LAYER, (double (*)[100])d_OUTPUT_WEIGHT, (double *)d_OUTPUT);
	cudaDeviceSynchronize();

	cudaMemcpy(OUTPUT, d_OUTPUT, sizeof(OUTPUT), cudaMemcpyDeviceToHost);

	cudaFree(d_THIRD_LAYER);
	cudaFree(d_OUTPUT_WEIGHT);
	cudaFree(d_OUTPUT);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Output layer: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}