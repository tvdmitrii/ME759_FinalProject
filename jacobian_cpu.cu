#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <fstream>
#include "f_eval.cuh"

using namespace std;

__inline__ __host__ __device__ double f_eval(double* p_x, int m);
double* readFile(string input, int *m, int *n);
void writeFile(string output, double* A, int m, int n);

int main(int argc, char* argv[])
{
	if(argc != 4){
		printf("USAGE: %s <inputArray.inp> <outfile> <epsilon>\n", argv[0]);
		return 1;
	}

	string inputFN = argv[1];
	string outputFN = argv[2];
	double epsilon = atof(argv[3]);
	int m, n;
	
	// Read file
	double* A = readFile(inputFN, &m, &n);
	double *B = (double*)malloc(sizeof(double)*m*n);
	
	struct timespec start, stop;
	double time = 0;
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			A[i*m+j] += epsilon;
			B[i*m+j] = f_eval(A+i*m,m);
			
			A[i*m+j] -= 2*epsilon;
			B[i*m+j] -= f_eval(A+i*m,m);
			
			A[i*m+j] += epsilon;
			B[i*m+j] /= 2*epsilon;
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &stop);
	time += (stop.tv_sec - start.tv_sec) * 1000;
    time += (stop.tv_nsec - start.tv_nsec) / 1000.0 / 1000.0;

	
	writeFile(outputFN, B, m, n);
	cout << n << "," << m << "," << time << endl;

	free(A);
	free(B);
	A = NULL;
	B = NULL;
}

double* readFile(string input, int *m, int *n){
	string line;
	ifstream in (input);
	if (in.is_open())
	{
		//read n
		getline(in,line);
		*n = atoi(line.c_str());
		
		//read m
		getline(in,line);
		*m = atoi(line.c_str());
		
		// Allocate pinned memory
		double *A = (double*)malloc(sizeof(double)*(*m)*(*n));
		
		for(int i = 0; i < *n; i++){
			for(int j = 0; j < *m-1; j++){
				getline(in,line,',');
				A[i*(*m)+j] = atof(line.c_str());
			}
			getline(in,line);
			A[(i+1)*(*m)-1] = atof(line.c_str());
		}
		
		in.close();
		
		return A;
	}
	else{
		*m = -1;
		*n = -1;
		cout << "Unable to open input file!"; 
		return NULL;
	}
}

void writeFile(string output, double* A, int m, int n){
	ofstream out (output);
	
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m-1; j++){
			out << A[i*m+j] << ",";
		}
		out << A[(i+1)*m-1] << endl;
	}
	
	out.close();
}