/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                    ( computation_kernels.cu )                      //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "computation.h"


__global__ void d_compute_first(double d_INPUT[29][29], double d_FIRST_LAYER_WEIGHT[6][13][13][5][5], double d_FIRST_LAYER[6][13][13]) { 
	int feature_map = blockIdx.x;
	int x = threadIdx.x;
	int y = threadIdx.y;

	double sum = 0;
	for(int i=0;i<5;++i) for(int j=0;j<5;++j)
		sum += d_FIRST_LAYER_WEIGHT[feature_map][x][y][i][j] * d_INPUT[2*x+i][2*y+j];

	d_FIRST_LAYER[feature_map][x][y] = SIGMOID(sum);
}


__global__ void d_compute_second(double d_FIRST_LAYER[6][13][13], double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5], double d_SECOND_LAYER[50][5][5]) { 
	int feature_map = blockIdx.x;
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	double sum = 0;
	for(int k=0;k<6;++k) for(int i=0;i<5;++i) for(int j=0;j<5;++j)
		sum += d_SECOND_LAYER_WEIGHT[feature_map][x][y][k][i][j] * d_FIRST_LAYER[k][2*x+i][2*y+j];

	d_SECOND_LAYER[feature_map][x][y] = SIGMOID(sum);
}


__global__ void d_compute_third(double d_SECOND_LAYER[50][5][5], double d_THIRD_LAYER_WEIGHT[100][50][5][5], double d_THIRD_LAYER[100]) { 
	__shared__ double patchy_sum[50];

	int x = blockIdx.x;
	int k = threadIdx.x; // nr of feature map of SECOND_LAYER

	patchy_sum[k] = 0;
	__syncthreads();
	for(int i=0;i<5;++i) for(int j=0;j<5;++j) {
		patchy_sum[k] += d_THIRD_LAYER_WEIGHT[x][k][i][j] * d_SECOND_LAYER[k][i][j];
	}

	__syncthreads();

	if(threadIdx.x == 0) {
		double sum = 0;
		for(int i=0;i<50;++i) sum += patchy_sum[i];

		d_THIRD_LAYER[x] = SIGMOID(sum);
	}
}


__global__ void d_compute_output(double d_THIRD_LAYER[100], double d_OUTPUT_WEIGHT[10][100], double d_OUTPUT[10]) { 
	__shared__ double shared_third_layer[100];

	int x = threadIdx.x;
	for(int i=0;i<10;++i) 
		shared_third_layer[10*x+i] = d_THIRD_LAYER[10*x+i];

	__syncthreads();

	double sum = 0;
	for(int i=0;i<100;++i) 
		sum += d_OUTPUT_WEIGHT[x][i] * shared_third_layer[i];

	d_OUTPUT[x] = SIGMOID(sum);
}