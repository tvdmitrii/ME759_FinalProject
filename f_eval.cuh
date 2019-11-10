

__inline__ __host__ __device__ double f_eval(double* p_x, int m){
	double result = 1;
	for(int i = 0; i < m; i++){
		result *= p_x[i];
	}
	
	return result;
}