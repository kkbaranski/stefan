/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                         ( computation.h )                          //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#ifndef COMPUTATION_H
#define COMPUTATION_H

#include "commons.h"
#include "network.h"

__host__ void compute_first();
__host__ void compute_second();
__host__ void compute_third();
__host__ void compute_output();

__global__ void d_compute_first(double d_INPUT[29][29], double d_FIRST_LAYER_WEIGHT[6][13][13][5][5], double d_FIRST_LAYER[6][13][13]);
__global__ void d_compute_second(double d_FIRST_LAYER[6][13][13], double d_SECOND_LAYER_WEIGHT[50][5][5][6][5][5], double d_SECOND_LAYER[50][5][5]);
__global__ void d_compute_third(double d_SECOND_LAYER[50][5][5], double d_THIRD_LAYER_WEIGHT[100][50][5][5], double d_THIRD_LAYER[100]);
__global__ void d_compute_output(double d_THIRD_LAYER[100], double d_OUTPUT_WEIGHT[10][100], double d_OUTPUT[10]);

#endif // COMPUTATION_H