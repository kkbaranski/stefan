/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                             ( main.cu )                            //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "network.h"

int main() {
	deserialize();

	time_t timer;
	char buffer[26];
	struct tm* tm_info;
	time(&timer);
	tm_info = localtime(&timer);
	strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", tm_info);
	printf("\n== %s ==\n", buffer);
	fprintf(stderr, "\n== %s ==\n", buffer);
	
	int CMD;


	int COUNTER = 0;
	int OK_COUNTER = 0;
	int FAIL_COUNTER = 0;


	while(true) {
		scanf("%d",&CMD);

		if(CMD == 0) 
			break;

		// test input
		if(CMD == 1) {
			load_input();
			compute();
			pprint_result();
		}

		// train
		if(CMD == 2) {
			int expected_digit;
			scanf("%d", &expected_digit);

			load_input();
			compute();
			int actual_digit = get_result();

			if(actual_digit == expected_digit) {
				OK_COUNTER++;
				printf(".");
			} else {
				FAIL_COUNTER++;
				printf("x");
				print_info(COUNTER, expected_digit);
				backpropagate(expected_digit);
				compute();
			}

			backpropagate(expected_digit);
			COUNTER++;
		}

		if((COUNTER%SERIALIZE_AFTER) == 0) {
			//serialize();
			printf("\n");
			fflush(stdout);
		}
	}

	printf("\nNumber of tests: %d, correct: %d (%.2lf%%), incorrect: %d (%.2lf%%), ETA: %.6lf\n", 
		COUNTER, 
		OK_COUNTER, 
		(100.0 * (double)OK_COUNTER / COUNTER), 
		FAIL_COUNTER, 
		(100.0 * (double)FAIL_COUNTER / COUNTER),
		ETA);

	serialize();
	return 0;
}
