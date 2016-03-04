/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                  ( backpropagation_kernels.cu )                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "backpropagation.h"


__global__ void d_backpropagate_first(	double d_FIRST_LAYER[6][13][13], 
																				double d_FIRST_LAYER_WEIGHT[6][13][13][5][5], 
																				double d_INPUT[29][29], 
																				double d_FIRST_LAYER_ERROR[6][13][13]) 
{ 
	int feature_map = blockIdx.x;		// [0..6)
	int x = blockIdx.y;							// [0..13)
	int y = blockIdx.z;							// [0..13)
	int i = threadIdx.x;						// [0..5)
	int j = threadIdx.y;						// [0..5)

	d_FIRST_LAYER_WEIGHT[feature_map][x][y][i][j] -= ETA * (d_INPUT[2*x+i][2*y+j] * DSIGMOID(d_FIRST_LAYER[feature_map][x][y]) * d_FIRST_LAYER_ERROR[feature_map][x][y]);
}


__global__ void d_backpropagate_second(	double d_SECOND_LAYER[50][5][5], 
																				double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5],
																				double d_SECOND_LAYER_ERROR[50][5][5],
																				double d_FIRST_LAYER_ERROR[6][13][13]) 
{
	__shared__ double patchy_error[50];

	int fm = blockIdx.x;						// [0..6)
	int x = blockIdx.y;							// [0..13)
	int y = blockIdx.z;							// [0..13)
	int k = threadIdx.x;						// [0..50)

	patchy_error[k] = 0;
	__syncthreads();
	for(int i=(int)floor((x-5)/2.0)+1; i<=x/2; ++i)
		for(int j=(int)floor((y-5)/2.0)+1; j<=y/2; ++j)
			if(i >= 0 && j >= 0 && i < 5 && j < 5)
				patchy_error[k] += d_SECOND_LAYER_WEIGHT[k][i][j][fm][x-2*i][y-2*j] * DSIGMOID(d_SECOND_LAYER[k][i][j]) * d_SECOND_LAYER_ERROR[k][i][j];
	__syncthreads();
	if(k == 0) {
		double sum_error = 0;
		for(int i=0;i<50;++i)
			sum_error += patchy_error[i];
		d_FIRST_LAYER_ERROR[fm][x][y] = sum_error;
	}
}


__global__ void d_update_second_layer_weights(	double d_SECOND_LAYER[50][5][5], 
																								double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5], 
																								double d_FIRST_LAYER[6][13][13], 
																								double d_SECOND_LAYER_ERROR[50][5][5]) 
{ 
	int fm = blockIdx.x;						// [0..50)
	int x = blockIdx.y;							// [0..5)
	int y = blockIdx.z;							// [0..5)
	int k = threadIdx.x;						// [0..6)
	int i = threadIdx.y;						// [0..5)
	int j = threadIdx.z;						// [0..5)

	d_SECOND_LAYER_WEIGHT[fm][x][y][k][i][j] -= ETA * (d_FIRST_LAYER[k][2*x+i][2*y+j] * DSIGMOID(d_SECOND_LAYER[fm][x][y]) * d_SECOND_LAYER_ERROR[fm][x][y]);
}


__global__ void d_backpropagate_third(	double d_THIRD_LAYER[100], 
																				double d_THIRD_LAYER_WEIGHT[100][50][5][5], 
																				double d_SECOND_LAYER[50][5][5], 
																				double d_THIRD_LAYER_ERROR[100], 
																				double d_SECOND_LAYER_ERROR[50][5][5]) 
{ 
	__shared__ double patchy_error[100];

	int k = threadIdx.x;						// nr of third layer neuron [0..100)
	int feature_map = blockIdx.x;		// [0..50)
	int x = blockIdx.y;							// [0..5)
	int y = blockIdx.z;							// [0..5)

	double q = DSIGMOID(d_THIRD_LAYER[k]) * d_THIRD_LAYER_ERROR[k];
	patchy_error[k] = d_THIRD_LAYER_WEIGHT[k][feature_map][x][y] * q;
	__syncthreads();
	if(k == 0) {
		double sum_error = 0;
		for(int i=0;i<100;++i)
			sum_error += patchy_error[i];
		d_SECOND_LAYER_ERROR[feature_map][x][y] = sum_error;
	}
	__syncthreads();

	d_THIRD_LAYER_WEIGHT[k][feature_map][x][y] -= ETA * (d_SECOND_LAYER[feature_map][x][y] * q);
}


__global__ void d_backpropagate_output(double d_OUTPUT[10], double d_OUTPUT_WEIGHT[10][100], double d_THIRD_LAYER[100], double d_OUTPUT_ERROR[10], double d_THIRD_LAYER_ERROR[100]) { 
	__shared__ double patchy_error[10];

	int x = threadIdx.x;	// nr of output neuron
	int y = blockIdx.x;		// nr of third layer neuron

	double q = DSIGMOID(d_OUTPUT[x]) * d_OUTPUT_ERROR[x];
	patchy_error[x] = d_OUTPUT_WEIGHT[x][y] * q;
	__syncthreads();
	if(x == 0) {
		double sum_error = 0;
		for(int i=0;i<10;++i) 
			sum_error += patchy_error[i];
		d_THIRD_LAYER_ERROR[y] = sum_error;
	}
	__syncthreads();

	d_OUTPUT_WEIGHT[x][y] -= ETA * (d_THIRD_LAYER[y] * q);
}