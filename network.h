/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                           ( network.h )                            //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#ifndef NETWORK_H
#define NETWORK_H

#include "commons.h"
#include "computation.h"
#include "backpropagation.h"

/* NEURAL NETWORK (3215 neurons) */
extern double INPUT[29][29];					// input layer (841 neurons)
extern double FIRST_LAYER[6][13][13];	// first convolutional layer (6 feature maps 13x13, 1014 neurons)
extern double SECOND_LAYER[50][5][5];	// second convolutional layer (50 feature maps 5x5, 1250 neurons)
extern double THIRD_LAYER[100];				// third fully-connected layer (100 neurons)
extern double OUTPUT[10];							// output layer (10 neurons)

/* DENDRITES WITH WEIGHTS */
extern double FIRST_LAYER_WEIGHT[6][13][13][5][5];			// 25350 weights
extern double SECOND_LAYER_WEIGHT[50][5][5][6][5][5];		// 187500 weights
extern double THIRD_LAYER_WEIGHT[100][50][5][5];				// 125000 weights
extern double OUTPUT_WEIGHT[10][100];										// 1000 weights

__host__ void compute();
__host__ void backpropagate(int expected_digit);

__host__ void load_input();
__host__ int get_result();

// printings
__host__ void print_input_layer();
__host__ void print_first_layer();
__host__ void print_second_layer();
__host__ void print_third_layer();
__host__ void print_output_layer();
__host__ void print_result();
__host__ void pprint_result();
__host__ void print_info(int nr, int expected_digit);

// serialization
__host__ void serialize();
__host__ void deserialize();
__host__ void generate();

#endif // NETWORK_H