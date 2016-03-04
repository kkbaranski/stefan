/*//////////////////////////////////////////////////////////////////////
//                                       _   __                       //
//                             _   __   / | / /                       //
//                   ___      / | / /  /  |/ /                        //
//                  /   |    /  |/ /  / /|  /                         //
//                 / /| |   / /|  /  /_/ |_/ E T W O R K              //
//                / ___ |  /_/ |_/ E  U  R  A  L                      //
//               /_/  |_| R T I F I C I A L                           //
//                                                                    //
//                          ( network.cu )                            //
//                                                                    //
////////////////////////////////////////////////////////////////////////
///-------------- Copyright © 2016 Krzysztof Baranski  --------------///
//////////////////////////////////////////////////////////////////////*/
#include "network.h"

/* NEURAL NETWORK (3215 neurons) */
double INPUT[29][29];						// input layer (841 neurons)
double FIRST_LAYER[6][13][13];	// first convolutional layer (6 feature maps 13x13, 1014 neurons)
double SECOND_LAYER[50][5][5];	// second convolutional layer (50 feature maps 5x5, 1250 neurons)
double THIRD_LAYER[100];				// third fully-connected layer (100 neurons)
double OUTPUT[10];							// output layer (10 neurons)

/* DENDRITES WITH WEIGHTS */
double FIRST_LAYER_WEIGHT[6][13][13][5][5];			// 25350 weights
double SECOND_LAYER_WEIGHT[50][5][5][6][5][5];	// 187500 weights
double THIRD_LAYER_WEIGHT[100][50][5][5];				// 125000 weights
double OUTPUT_WEIGHT[10][100];									// 1000 weights


__host__ void compute() {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	compute_first();
	compute_second();
	compute_third();
	compute_output();

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Computation time: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}

__host__ void backpropagate(int expected_digit) {
	// struct timeval start, stop, diff;
	// gettimeofday(&start, NULL);

	calculate_output_error(expected_digit);

	backpropagate_output();
	backpropagate_third();
	backpropagate_second();
	backpropagate_first();

	// gettimeofday(&stop, NULL);
	// timersub(&stop, &start, &diff);
	// fprintf(stderr, "Backpropagation time: %ld.%06lds\n", (long int)diff.tv_sec, (long int)diff.tv_usec);
}


__host__ void load_input() {
	for(int a=0;a<29;++a) 
		for(int b=0;b<29;++b)
			scanf("%lf", &INPUT[a][b]);
}

__host__ int get_result() {
	int best_digit = 0;
	for(int i=0; i<10; ++i) 
		if(OUTPUT[i] > OUTPUT[best_digit]) 
			best_digit = i;
	return best_digit;
}


/* PRINTINGS */
__host__ void print_input_layer() {
	//printf("=== INPUT ===\n");
	for(int a=0;a<29;++a,printf("\n")) 
		for(int b=0;b<29;++b)
			printf("%lf ",INPUT[a][b]);
}

__host__ void print_first_layer() {
	//printf("=== FIRST LAYER ===\n");
	for(int a=0;a<6;++a,printf("\n")) 
		for(int b=0;b<13;++b,printf("\n"))
			for(int c=0;c<13;++c)
				printf("%lf ",FIRST_LAYER[a][b][c]);
}

__host__ void print_second_layer() {
	//printf("=== SECOND LAYER ===\n");
	for(int a=0;a<50;++a,printf("\n")) 
		for(int b=0;b<5;++b,printf("\n"))
			for(int c=0;c<5;++c)
				printf("%lf ",SECOND_LAYER[a][b][c]);
}

__host__ void print_third_layer() {
	//printf("=== THIRD LAYER ===\n");
	for(int a=0;a<100;++a)
		printf("%lf ",THIRD_LAYER[a]);
	printf("\n");
}

__host__ void print_output_layer() {
	//printf("=== OUTPUT ===\n");
	for(int a=0;a<10;++a)
		printf("%lf ",OUTPUT[a]);
	printf("\n");
}

__host__ void print_result() {
	int best_digit = 0;
	for(int i=0; i<10; ++i) 
		if(OUTPUT[i] > OUTPUT[best_digit]) 
			best_digit = i;
	printf("%d\n", best_digit);
}

__host__ void pprint_result() {
	int best_digit = 0;
	for(int i=0; i<10; ++i) 
		if(OUTPUT[i] > OUTPUT[best_digit]) 
			best_digit = i;
	printf("#########\n");
	printf("#       #\n");
	printf("#   %d   #\n", best_digit);
	printf("#       #\n");
	printf("#########\n");
	for(int i=0; i<10; ++i)
		printf("%s %d: % .10lf\n", (i == best_digit ? ">" : " "), i, OUTPUT[i]);
}

__host__ void print_info(int nr, int expected_digit) {
	int actual_digit = get_result();
	fprintf(stderr, ">>> TEST nr: %d\n", nr);
	fprintf(stderr, "    Actual:   %d\n", actual_digit);
	fprintf(stderr, "    Expected: %d\n", expected_digit);
	for(int i=0; i<10; ++i)
		fprintf(stderr, "      %s %d: % .10lf\n", (i == actual_digit ? ">" : (i == expected_digit ? "?":" ")), i, OUTPUT[i]);
}


/* SERIALIZATION */
__host__ void serialize() {
	FILE *fp = fopen(NETWORK_FILE, "w");
	// FIRST_LAYER_WEIGHT
	for(int a=0;a<6;++a,fprintf(fp,"\n")) 
		for(int b=0;b<13;++b,fprintf(fp,"\n"))
			for(int c=0;c<13;++c,fprintf(fp,"\n"))
				for(int d=0;d<5;++d,fprintf(fp,"\n"))
					for(int e=0;e<5;++e)
						fprintf(fp, "%.20lf ",FIRST_LAYER_WEIGHT[a][b][c][d][e]);
	// SECOND_LAYER_WEIGHT
	for(int a=0;a<50;++a,fprintf(fp,"\n")) 
		for(int b=0;b<5;++b,fprintf(fp,"\n"))
			for(int c=0;c<5;++c,fprintf(fp,"\n"))
				for(int d=0;d<6;++d,fprintf(fp,"\n"))
					for(int e=0;e<5;++e,fprintf(fp,"\n"))
						for(int f=0;f<5;++f)
							fprintf(fp, "%.20lf ",SECOND_LAYER_WEIGHT[a][b][c][d][e][f]);
	// THIRD_LAYER_WEIGHT
	for(int a=0;a<100;++a,fprintf(fp,"\n")) 
		for(int b=0;b<50;++b,fprintf(fp,"\n"))
			for(int c=0;c<5;++c,fprintf(fp,"\n"))
				for(int d=0;d<5;++d)
					fprintf(fp, "%.20lf ",THIRD_LAYER_WEIGHT[a][b][c][d]);
	// OUTPUT_WEIGHT
	for(int a=0;a<10;++a,fprintf(fp,"\n"))
		for(int b=0;b<100;++b)
			fprintf(fp, "%.20lf ",OUTPUT_WEIGHT[a][b]);
	fclose(fp);
}


__host__ void deserialize() {
	FILE *fp = fopen(NETWORK_FILE, "r");
	if(fp == NULL) {
		fprintf(stderr, "File '%s' does not exist!\n", NETWORK_FILE);
		generate();
		return;
	}
	// FIRST_LAYER_WEIGHT
	for(int a=0;a<6;++a) 
		for(int b=0;b<13;++b)
			for(int c=0;c<13;++c)
				for(int d=0;d<5;++d)
					for(int e=0;e<5;++e)
						fscanf(fp, "%lf", &FIRST_LAYER_WEIGHT[a][b][c][d][e]);
	// SECOND_LAYER_WEIGHT
	for(int a=0;a<50;++a) 
		for(int b=0;b<5;++b)
			for(int c=0;c<5;++c)
				for(int d=0;d<6;++d)
					for(int e=0;e<5;++e)
						for(int f=0;f<5;++f)
							fscanf(fp, "%lf", &SECOND_LAYER_WEIGHT[a][b][c][d][e][f]);
	// THIRD_LAYER_WEIGHT
	for(int a=0;a<100;++a) 
		for(int b=0;b<50;++b)
			for(int c=0;c<5;++c)
				for(int d=0;d<5;++d)
					fscanf(fp, "%lf", &THIRD_LAYER_WEIGHT[a][b][c][d]);
	// OUTPUT_WEIGHT
	for(int a=0;a<10;++a)
		for(int b=0;b<100;++b)
			fscanf(fp, "%lf", &OUTPUT_WEIGHT[a][b]);
	fclose(fp);
}


__host__ void generate() {
	srand(time(NULL));
	// FIRST_LAYER_WEIGHT
	for(int a=0;a<6;++a) 
		for(int b=0;b<13;++b)
			for(int c=0;c<13;++c)
				for(int d=0;d<5;++d)
					for(int e=0;e<5;++e)
						//FIRST_LAYER_WEIGHT[a][b][c][d][e] = -0.001;
						FIRST_LAYER_WEIGHT[a][b][c][d][e] = ((double)rand()/(double)RAND_MAX)-0.5;
	// SECOND_LAYER_WEIGHT
	for(int a=0;a<50;++a) 
		for(int b=0;b<5;++b)
			for(int c=0;c<5;++c)
				for(int d=0;d<6;++d)
					for(int e=0;e<5;++e)
						for(int f=0;f<5;++f)
							//SECOND_LAYER_WEIGHT[a][b][c][d][e][f] = -0.001;
							SECOND_LAYER_WEIGHT[a][b][c][d][e][f] = ((double)rand()/(double)RAND_MAX)-0.5;
	// THIRD_LAYER_WEIGHT
	for(int a=0;a<100;++a) 
		for(int b=0;b<50;++b)
			for(int c=0;c<5;++c)
				for(int d=0;d<5;++d)
					//THIRD_LAYER_WEIGHT[a][b][c][d] = -0.001;
					THIRD_LAYER_WEIGHT[a][b][c][d] = ((double)rand()/(double)RAND_MAX)-0.5;
	// OUTPUT_WEIGHT
	for(int a=0;a<10;++a)
		for(int b=0;b<100;++b)
			//OUTPUT_WEIGHT[a][b] = -0.001;
			OUTPUT_WEIGHT[a][b] = ((double)rand()/(double)RAND_MAX)-0.5;
}
