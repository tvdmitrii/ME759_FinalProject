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

__global__ void jacobian(double *dA, double *dB, int m, int step, int n, double eps)
{
	extern __shared__ double p[];
	double *f = (double*)&p[step];
	// Local line index
	int lli = threadIdx.x*m;
	// Global line index
	int gli = blockIdx.x*step + lli;
	
	if(gli < n*m){
	// Load a block into shared memory
	for(int i = 0; i < m; i++){
		p[lli + i] = dA[gli + i];
	}
	__threadfence();
	
	// Every thread computes partial derivatives at a single point
	for(int i = 0; i < m; i++){
		p[lli + i] += eps;
		__threadfence();
		// f(x1,x2,...,xi+eps,...,xm)
		f[lli + i] = f_eval(p+lli, m);
		
		p[lli + i] -= 2*eps;
		__threadfence();
		// f(x1,x2,...,xi-eps,...,xm)
		f[lli + i] -= f_eval(p+lli, m);
		
		p[lli + i] += eps;
		__threadfence();
		// df/dxi = (f(x1,x2,...,xi+eps,...,xm) - f(x1,x2,...,xi-eps,...,xm))/2/eps
		dB[gli+i] = f[lli + i]/eps/2;
	}
	}
}

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
	
	cudaEvent_t start_in, stop_in, start_ex, stop_ex;
	cudaEventCreate(&start_in);
	cudaEventCreate(&stop_in);
	cudaEventCreate(&start_ex);
	cudaEventCreate(&stop_ex);
	
	// Read file
	double* A = readFile(inputFN, &m, &n);
	size_t size = sizeof(double)*m*n;
	
	// Allocate memory on the device
	double *dA, *dB;
	cudaMalloc(&dA, size);
	cudaMalloc(&dB, size);
	
	int N_blocks, N_threads;
	// Shared memory is the limiting factor
	// Calculate how many points would fit
	// Division by 2 is because function values are also stored in shared memory
	int a = 48*1024/8/m/2;
	if(a >= n){
		N_threads = n;
		N_blocks = 1;
	} else {
		N_threads = a;
		N_blocks = n/N_threads+1;
	}
	
	//Start inclusive timer
	cudaEventRecord(start_in, 0);
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
	
	//Start exclusive timer
	cudaEventRecord(start_ex, 0);
	
	jacobian<<<N_blocks, N_threads, sizeof(double)*2*m*N_threads>>>(dA, dB, m, N_threads*m, n, epsilon);
	cudaDeviceSynchronize();
	
	//Stop exclusive timer
	cudaEventRecord(stop_ex, 0);
	cudaEventSynchronize(stop_ex);
	
	cudaMemcpy(A, dB, size, cudaMemcpyDeviceToHost);
	
	//Stop inclusive timer
	cudaEventRecord(stop_in, 0);
	cudaEventSynchronize(stop_in);
	
	float time_in, time_ex;
	cudaEventElapsedTime(&time_in, start_in, stop_in);
	cudaEventElapsedTime(&time_ex, start_ex, stop_ex);
	
	writeFile(outputFN, A, m, n);
	cout << n << "," << m << "," << time_ex << "," << time_in << endl;

	cudaFree(A);
	cudaFree(dA);
	cudaFree(dB);
	A = NULL;
	dA = NULL;
	dB = NULL;
	cudaEventDestroy(start_in);
	cudaEventDestroy(stop_in);
	cudaEventDestroy(start_ex);
	cudaEventDestroy(stop_ex);
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
		double *A;
		cudaMallocHost(&A, sizeof(double)*(*m)*(*n));
		
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