/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                      ( backpropagation.cu )                        //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "backpropagation.h"

/* ERRORS */
double FIRST_LAYER_ERROR[6][13][13];
double SECOND_LAYER_ERROR[50][5][5];
double THIRD_LAYER_ERROR[100];
double OUTPUT_ERROR[10];


__host__ void calculate_output_error(int expected_digit) {
	for(int i=0;i<10;++i)
		if(i == expected_digit)
			OUTPUT_ERROR[i] = OUTPUT[i] - 1;
		else
			OUTPUT_ERROR[i] = OUTPUT[i] + 1;
}


__host__ void backpropagate_first() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_FIRST_LAYER;
	double* d_FIRST_LAYER_WEIGHT;
	double* d_INPUT;
	double* d_FIRST_LAYER_ERROR;
	
	cudaMalloc(&d_FIRST_LAYER, sizeof(FIRST_LAYER));
	cudaMalloc(&d_FIRST_LAYER_WEIGHT, sizeof(FIRST_LAYER_WEIGHT));
	cudaMalloc(&d_INPUT, sizeof(INPUT));
	cudaMalloc(&d_FIRST_LAYER_ERROR, sizeof(FIRST_LAYER_ERROR));

	cudaMemcpy(d_FIRST_LAYER, FIRST_LAYER, sizeof(FIRST_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FIRST_LAYER_WEIGHT, FIRST_LAYER_WEIGHT, sizeof(FIRST_LAYER_WEIGHT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_INPUT, INPUT, sizeof(INPUT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FIRST_LAYER_ERROR, FIRST_LAYER_ERROR, sizeof(FIRST_LAYER_ERROR), cudaMemcpyHostToDevice);

	dim3 numBlocks(6, 13, 13);
	dim3 threadsPerBlock(5, 5);
	d_backpropagate_first<<<numBlocks, threadsPerBlock>>>(	(double (*)[13][13]) 				d_FIRST_LAYER,
																													(double (*)[13][13][5][5]) 	d_FIRST_LAYER_WEIGHT,
																													(double (*)[29]) 						d_INPUT,
																													(double (*)[13][13]) 				d_FIRST_LAYER_ERROR);
	cudaDeviceSynchronize();

	cudaMemcpy(FIRST_LAYER_WEIGHT, d_FIRST_LAYER_WEIGHT, sizeof(FIRST_LAYER_WEIGHT), cudaMemcpyDeviceToHost);

	cudaFree(d_FIRST_LAYER);
	cudaFree(d_FIRST_LAYER_WEIGHT);
	cudaFree(d_INPUT);
	cudaFree(d_FIRST_LAYER_ERROR);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "First layer backpropagation: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}


__host__ void backpropagate_second() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_SECOND_LAYER;
	double* d_SECOND_LAYER_WEIGHT;
	double* d_FIRST_LAYER;
	double* d_SECOND_LAYER_ERROR;
	double* d_FIRST_LAYER_ERROR;

	cudaMalloc(&d_SECOND_LAYER, sizeof(SECOND_LAYER));
	cudaMalloc(&d_SECOND_LAYER_WEIGHT, sizeof(SECOND_LAYER_WEIGHT));
	cudaMalloc(&d_FIRST_LAYER, sizeof(FIRST_LAYER));
	cudaMalloc(&d_SECOND_LAYER_ERROR, sizeof(SECOND_LAYER_ERROR));
	cudaMalloc(&d_FIRST_LAYER_ERROR, sizeof(FIRST_LAYER_ERROR));

	cudaMemcpy(d_SECOND_LAYER, SECOND_LAYER, sizeof(SECOND_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SECOND_LAYER_WEIGHT, SECOND_LAYER_WEIGHT, sizeof(SECOND_LAYER_WEIGHT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FIRST_LAYER, FIRST_LAYER, sizeof(FIRST_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SECOND_LAYER_ERROR, SECOND_LAYER_ERROR, sizeof(SECOND_LAYER_ERROR), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FIRST_LAYER_ERROR, FIRST_LAYER_ERROR, sizeof(FIRST_LAYER_ERROR), cudaMemcpyHostToDevice);

	// CALCULATE FIRST_LAYER_ERROR
	dim3 numBlocks(6, 13, 13);
	dim3 threadsPerBlock(50);
	d_backpropagate_second<<<numBlocks, threadsPerBlock>>>(	(double (*)[5][5]) 					d_SECOND_LAYER,
																													(double (*)[5][5][6][5][5]) d_SECOND_LAYER_WEIGHT,
																													(double (*)[5][5]) 					d_SECOND_LAYER_ERROR,
																													(double (*)[13][13])				d_FIRST_LAYER_ERROR);

	// UPDATE WEIGHTS
	dim3 numBlocks2(50, 5, 5);
	dim3 threadsPerBlock2(6, 5, 5);
	d_update_second_layer_weights<<<numBlocks2, threadsPerBlock2>>>(	(double (*)[5][5]) 					d_SECOND_LAYER,
																																		(double (*)[5][5][6][5][5]) d_SECOND_LAYER_WEIGHT,
																																		(double (*)[13][13]) 				d_FIRST_LAYER,
																																		(double (*)[5][5]) 					d_SECOND_LAYER_ERROR);
	cudaDeviceSynchronize();

	cudaMemcpy(SECOND_LAYER_WEIGHT, d_SECOND_LAYER_WEIGHT, sizeof(SECOND_LAYER_WEIGHT), cudaMemcpyDeviceToHost);
	cudaMemcpy(FIRST_LAYER_ERROR, d_FIRST_LAYER_ERROR, sizeof(FIRST_LAYER_ERROR), cudaMemcpyDeviceToHost);

	cudaFree(d_SECOND_LAYER);
	cudaFree(d_SECOND_LAYER_WEIGHT);
	cudaFree(d_FIRST_LAYER);
	cudaFree(d_SECOND_LAYER_ERROR);
	cudaFree(d_FIRST_LAYER_ERROR);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Second layer backpropagation: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}


__host__ void backpropagate_third() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_THIRD_LAYER;
	double* d_THIRD_LAYER_WEIGHT;
	double* d_SECOND_LAYER;
	double* d_THIRD_LAYER_ERROR;
	double* d_SECOND_LAYER_ERROR;
	
	cudaMalloc(&d_THIRD_LAYER, sizeof(THIRD_LAYER));
	cudaMalloc(&d_THIRD_LAYER_WEIGHT, sizeof(THIRD_LAYER_WEIGHT));
	cudaMalloc(&d_SECOND_LAYER, sizeof(SECOND_LAYER));
	cudaMalloc(&d_THIRD_LAYER_ERROR, sizeof(THIRD_LAYER_ERROR));
	cudaMalloc(&d_SECOND_LAYER_ERROR, sizeof(SECOND_LAYER_ERROR));

	cudaMemcpy(d_THIRD_LAYER, THIRD_LAYER, sizeof(THIRD_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_THIRD_LAYER_WEIGHT, THIRD_LAYER_WEIGHT, sizeof(THIRD_LAYER_WEIGHT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SECOND_LAYER, SECOND_LAYER, sizeof(SECOND_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_THIRD_LAYER_ERROR, THIRD_LAYER_ERROR, sizeof(THIRD_LAYER_ERROR), cudaMemcpyHostToDevice);

	dim3 numBlocks(50, 5, 5);
	int threadsPerBlock = 100;
	d_backpropagate_third<<<numBlocks, threadsPerBlock>>>(	(double *) 							d_THIRD_LAYER,
																													(double (*)[50][5][5]) 	d_THIRD_LAYER_WEIGHT,
																													(double (*)[5][5]) 			d_SECOND_LAYER,
																													(double *) 							d_THIRD_LAYER_ERROR,
																													(double (*)[5][5]) 			d_SECOND_LAYER_ERROR);
	cudaDeviceSynchronize();

	cudaMemcpy(THIRD_LAYER_WEIGHT, d_THIRD_LAYER_WEIGHT, sizeof(THIRD_LAYER_WEIGHT), cudaMemcpyDeviceToHost);
	cudaMemcpy(SECOND_LAYER_ERROR, d_SECOND_LAYER_ERROR, sizeof(SECOND_LAYER_ERROR), cudaMemcpyDeviceToHost);

	cudaFree(d_THIRD_LAYER);
	cudaFree(d_THIRD_LAYER_WEIGHT);
	cudaFree(d_SECOND_LAYER);
	cudaFree(d_THIRD_LAYER_ERROR);
	cudaFree(d_SECOND_LAYER_ERROR);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Third layer backpropagation: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}


__host__ void backpropagate_output() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	double* d_OUTPUT;
	double* d_OUTPUT_WEIGHT;
	double* d_THIRD_LAYER;
	double* d_OUTPUT_ERROR;
	double* d_THIRD_LAYER_ERROR;
	
	cudaMalloc(&d_OUTPUT, sizeof(OUTPUT));
	cudaMalloc(&d_OUTPUT_WEIGHT, sizeof(OUTPUT_WEIGHT));
	cudaMalloc(&d_THIRD_LAYER, sizeof(THIRD_LAYER));
	cudaMalloc(&d_OUTPUT_ERROR, sizeof(OUTPUT_ERROR));
	cudaMalloc(&d_THIRD_LAYER_ERROR, sizeof(THIRD_LAYER_ERROR));

	cudaMemcpy(d_OUTPUT, OUTPUT, sizeof(OUTPUT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OUTPUT_WEIGHT, OUTPUT_WEIGHT, sizeof(OUTPUT_WEIGHT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_THIRD_LAYER, THIRD_LAYER, sizeof(THIRD_LAYER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_OUTPUT_ERROR, OUTPUT_ERROR, sizeof(OUTPUT_ERROR), cudaMemcpyHostToDevice);

	int numBlocks = 100;
	int threadsPerBlock = 10;
	d_backpropagate_output<<<numBlocks, threadsPerBlock>>>(	(double *) 					d_OUTPUT,
																													(double (*)[100]) 	d_OUTPUT_WEIGHT,
																													(double *) 					d_THIRD_LAYER,
																													(double *) 					d_OUTPUT_ERROR,
																													(double *) 					d_THIRD_LAYER_ERROR);
	cudaDeviceSynchronize();

	cudaMemcpy(OUTPUT_WEIGHT, d_OUTPUT_WEIGHT, sizeof(OUTPUT_WEIGHT), cudaMemcpyDeviceToHost);
	cudaMemcpy(THIRD_LAYER_ERROR, d_THIRD_LAYER_ERROR, sizeof(THIRD_LAYER_ERROR), cudaMemcpyDeviceToHost);

	cudaFree(d_OUTPUT);
	cudaFree(d_OUTPUT_WEIGHT);
	cudaFree(d_THIRD_LAYER);
	cudaFree(d_OUTPUT_ERROR);
	cudaFree(d_THIRD_LAYER_ERROR);

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Output layer backpropagation: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}
