#pragma once

#include "BasicKernel.h"

/// <summary>
/// ���������� GPU
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_GPUKernel(ValueType* pVector,int nVectorSize, int nBlockSize, sycl::queue* pstDPCQueue, ValueType& _OutResult);

/// <summary>
/// ���������� CPU
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_CPUKernel(ValueType* pVector, int nVectorSize, sycl::queue* pstDPCQueue, ValueType& _OutResult);

/// <summary>
/// ���������� AVX256
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_AVXKernel(ValueType* pVector, int nVectorSize, int nBlockSize, sycl::queue* pstDPCQueue);
