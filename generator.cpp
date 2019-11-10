#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

void writeFile(string output, double* A, int m, int n){
	ofstream out (output);
	
	out << n << endl;
	out << m << endl;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m-1; j++){
			out << A[i*m+j] << ",";
		}
		out << A[(i+1)*m-1] << endl;
	}
	
	out.close();
}

int main(int argc, char* argv[])
{
	if(argc != 3){
		printf("USAGE: %s <n> <m>\n", argv[0]);
		return 1;
	}
	
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	
	double* A = (double*) malloc(sizeof(double)*m*n);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			A[i*m+j] = i;
		}
	}
	
	writeFile("inputArray" + to_string(n) + "-" + to_string(m) + ".inp", A, m, n);
	
	free(A);
	A = NULL;
}