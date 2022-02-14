/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iterator>
#include <iomanip>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
using namespace std;


/**
 * CiiUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorQuantumComputing(float *qbit_input_quantum_state, float *qbit_quantum_gate, float *qbit_output_quantum_state,int qbit_value,  int numElements)
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i < numElements && i % int (__powf(2,(qbit_value+1)))<int (__powf(2,(qbit_value))))
        {
            qbit_output_quantum_state[i] = (qbit_quantum_gate[0] * qbit_input_quantum_state[i] + qbit_quantum_gate[1] * qbit_input_quantum_state[i | (1 << qbit_value)]);
            qbit_output_quantum_state[i | (1 << qbit_value)] = (qbit_quantum_gate[2] * qbit_input_quantum_state[i] + qbit_quantum_gate[3] * qbit_input_quantum_state[i | (1 << qbit_value)]);

        }
    
}

/**
 * Host main routine
 */
int
main(void)
{


    FILE * FP;
    FP=fopen("input.txt","r");
    int number_of_lines;
    char element;
    float qbit_quantum_gate_temp[2][2];


    if(FP==NULL){
        cout<<"File not found"<<endl;
        return 0;
    }

    while (EOF != (element=getc(FP))) {
        if ('\n' == element)
            number_of_lines=number_of_lines+1;
    }

    float* qbit_input_quantum_state = new float [(number_of_lines-3)];
    float* qbit_output_quantum_state = new float [(number_of_lines-4)];

    FP=fopen("input.txt","r");

    int i=0;
    while(fscanf(FP, "%f %f", &qbit_quantum_gate_temp[i][0], &qbit_quantum_gate_temp[i][1]) != EOF)
    {
        i++;
        if (i>1)
        {
            i = 0;
            while (fscanf(FP, "%f ", &qbit_input_quantum_state[i]) != EOF)
            {
                i++;
            }
            break;
        }
    }



    int qbit_input_quantum_state_length=number_of_lines-4;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = number_of_lines-4;
    size_t size = numElements  * sizeof(float);
    size_t size_gate = 4 * sizeof(float);
    size_t size_out = numElements * sizeof(float);



int qbit_value=qbit_input_quantum_state[numElements];
//cout<<"last element is  "<<qbit_input_quantum_state[numElements]<<endl;
//cout<<"Number of lines "<<number_of_lines<<endl;
   
 // Allocate the host input vector A

    // Allocate the host input vector B
    float* qbit_quantum_gate=new float [4];
    qbit_quantum_gate[0]=qbit_quantum_gate_temp[0][0];
    qbit_quantum_gate[1]=qbit_quantum_gate_temp[0][1];
    qbit_quantum_gate[2]=qbit_quantum_gate_temp[1][0];
    qbit_quantum_gate[3]=qbit_quantum_gate_temp[1][1];
    // Allocate the host output vector C
    // Verify that allocations succeeded
    if (qbit_input_quantum_state == NULL || qbit_quantum_gate == NULL || qbit_output_quantum_state == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    // Initialize the host input vectors
    // Allocate the device input vector A
    float *d_qbit_input_quantum_state = NULL;
    err = cudaMalloc((void **)&d_qbit_input_quantum_state, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_qbit_input_quantum_state  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate the device input vector B
    float *d_qbit_quantum_gate = NULL;
    err = cudaMalloc((void **)&d_qbit_quantum_gate, size_gate);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_qbit_quantum_gate  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_qbit_output_quantum_state = NULL;
    err = cudaMalloc((void **)&d_qbit_output_quantum_state, size_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_qbit_output_quantum_state (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_qbit_input_quantum_state, qbit_input_quantum_state, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit_input_quantum_state from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_qbit_quantum_gate, qbit_quantum_gate, size_gate, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit_quantum_gate from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorQuantumComputing<<<blocksPerGrid, threadsPerBlock>>>(d_qbit_input_quantum_state, d_qbit_quantum_gate, d_qbit_output_quantum_state,qbit_value, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(qbit_output_quantum_state, d_qbit_output_quantum_state, size_out, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit_output_quantum_state from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

 //   for(i=0;i<4;i++)
   // {
     //   cout<<fixed<<setprecision(3)<<qbit_quantum_gate[i]<<endl;
//}


//cout<<"---------------------------------------------------"<<endl;
  //  for(i=0;i<=qbit_input_quantum_state_length;i++)
   // {
     //   cout<<fixed<<setprecision(3)<<qbit_input_quantum_state[i]<<endl;
//}
//cout<<"---------------------------------------------------"<<endl;
    for(i=0;i<qbit_input_quantum_state_length;i++)
    {
        cout<<fixed<<setprecision(3)<<qbit_output_quantum_state[i]<<endl;
    }
    // Verify that the result vector is correct
    // Free device global memory
    err = cudaFree(d_qbit_input_quantum_state);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_qbit_input_quantum_state  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_qbit_quantum_gate);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_qbit_quantum_gate (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_qbit_output_quantum_state);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_qbit_output_quantum_state  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free host memory
    free(qbit_input_quantum_state);
    free(qbit_quantum_gate);
    free(qbit_output_quantum_state);


    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

