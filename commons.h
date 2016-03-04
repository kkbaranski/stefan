/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                           ( commons.h )                            //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#ifndef COMMONS_H
#define COMMONS_H

// includes
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

// functions
#define SIGMOID(x) (tanh(x))
#define DSIGMOID(x) (1-(x)*(x))

// consts
#define NETWORK_FILE ".network"
#define SERIALIZE_AFTER 100

const double ETA = 0.00005;

#endif // COMMONS_H
