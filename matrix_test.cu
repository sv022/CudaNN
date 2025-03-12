// void test_matrix(){
//     const int n = 4;
//     const int m = 7;
//     const int p = 3;
//     // float *test_a = (float*)malloc(sizeof(float) * n * m);
//     // float *test_b = (float*)malloc(sizeof(float) * n * m);
//     // float *test_c = (float*)malloc(sizeof(float) * n * n);

//     float test_a[n * m];
// 	Matrix::initRandomi_static(test_a, n, m);

// 	float test_b[m * n];
// 	Matrix::initRandomi_static(test_b, m, p);

// 	float test_c[n * p];
// 	Matrix::initRandomi_static(test_c, n, p);

//     float *d_test_a = 0;
// 	float *d_test_b = 0;
// 	float *d_test_c = 0;

// 	cudaMalloc(&d_test_a, sizeof(test_a));
//  	cudaMalloc(&d_test_b, sizeof(test_b));
//  	cudaMalloc(&d_test_c, sizeof(test_c));

//     Matrix::initRandomf_static(test_a, n, m);
//     Matrix::initRandomf_static(test_b, m, p);
//     Matrix::init_static(test_c, n, p);

//     Matrix::log_static(test_a, n, m);
//     Matrix::log_static(test_b, m, p);

//     cudaMemcpy(
//         d_test_a,
//         test_a,
//         sizeof(test_a),
//         cudaMemcpyHostToDevice
//     );
//     cudaMemcpy(
//         d_test_b,
//         test_b,
//         sizeof(test_b),
//         cudaMemcpyHostToDevice
//     );
//     cudaMemcpy(
//         d_test_c,
//         test_c,
//         sizeof(test_c),
//         cudaMemcpyHostToDevice
//     );

//     dim3 THREADS;
//  	THREADS.x = 16;
//  	THREADS.y = 16;
 
//  	int blocks = (n + THREADS.x - 1) / THREADS.x;
 
//  	dim3 BLOCKS;
//  	BLOCKS.x = blocks;
//  	BLOCKS.y = blocks;

//     Kernel::dot<<<BLOCKS, THREADS>>>(d_test_a, d_test_b, d_test_c, n, m, p);

//     cudaMemcpy(
//         test_c,
//         d_test_c,
//         sizeof(test_c),
//         cudaMemcpyDeviceToHost
//     );

//     Matrix::log_static(test_c, n, p);

    
//     float test_c_T[p * n];
// 	Matrix::init_static(test_c_T, p, n);
    
//     float *d_test_c_T = 0;
// 	cudaMalloc(&d_test_c_T, sizeof(float) * p * n);
    
//     cudaMemcpy(
//         d_test_c_T,
//         test_c_T,
//         sizeof(float) * p * n,
//         cudaMemcpyHostToDevice
//     );
    
//     cudaFree(d_test_c);
//  	cudaMalloc(&d_test_c, sizeof(float) * n * p);

//     cudaMemcpy(
//         d_test_c,
//         test_c,
//         sizeof(test_c),
//         cudaMemcpyHostToDevice
//     );

//     Kernel::transpose<<<BLOCKS, THREADS>>>(d_test_c, d_test_c_T, n, p);

//     cudaMemcpy(
//         test_c_T,
//         d_test_c_T,
//         sizeof(float) * p * n,
//         cudaMemcpyDeviceToHost
//     );

//     Matrix::log_static(test_c_T, p, n);


//     cudaFree(d_test_a);
//     cudaFree(d_test_b);
//     cudaFree(d_test_c);
//     cudaFree(d_test_c_T);
// }

