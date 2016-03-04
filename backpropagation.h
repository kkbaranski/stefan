/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                       ( backpropagation.h )                        //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "commons.h"
#include "network.h"

/* ERRORS */
extern double FIRST_LAYER_ERROR[6][13][13];
extern double SECOND_LAYER_ERROR[50][5][5];
extern double THIRD_LAYER_ERROR[100];
extern double OUTPUT_ERROR[10];

__host__ void calculate_output_error(int expected_digit);

__host__ void backpropagate_first();
__host__ void backpropagate_second();
__host__ void backpropagate_third();
__host__ void backpropagate_output();

__global__ void d_backpropagate_first(double d_FIRST_LAYER[6][13][13], double d_FIRST_LAYER_WEIGHT[6][13][13][5][5], double d_INPUT[29][29], double d_FIRST_LAYER_ERROR[6][13][13]);
__global__ void d_backpropagate_second(double d_SECOND_LAYER[50][5][5], double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5], double d_SECOND_LAYER_ERROR[50][5][5], double d_FIRST_LAYER_ERROR[6][13][13]);
__global__ void d_update_second_layer_weights(double d_SECOND_LAYER[50][5][5], double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5], double d_FIRST_LAYER[6][13][13], double d_SECOND_LAYER_ERROR[50][5][5]);
__global__ void d_backpropagate_third(double d_THIRD_LAYER[100], double d_THIRD_LAYER_WEIGHT[100][50][5][5], double d_SECOND_LAYER[50][5][5], double d_THIRD_LAYER_ERROR[100], double d_SECOND_LAYER_ERROR[50][5][5]);
__global__ void d_backpropagate_output(double d_OUTPUT[10], double d_OUTPUT_WEIGHT[10][100], double d_THIRD_LAYER[100], double d_OUTPUT_ERROR[10], double d_THIRD_LAYER_ERROR[100]);

#endif // BACKPROPAGATION_H