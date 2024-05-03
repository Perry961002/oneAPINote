#pragma once

#include "BasicKernel.h"

/// <summary>
/// 矩阵乘法的核函数
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">输入矩阵A</param>
/// <param name="pMatrixB">输入矩阵B</param>
/// <param name="pMatrixC">输出矩阵C</param>
/// <param name="nMatrixShapeM">矩阵C的行数</param>
/// <param name="nMatrixShapeN">矩阵C的列数</param>
/// <param name="nMatrixShapeK">矩阵A的行数</param>
/// <param name="nBlockSize">数据划分的块大小</param>
/// <param name="pstDPCQueue">DPC++的设备队列</param>
/// <returns></returns>
int MatrixMulti_GPUKernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// 矩阵乘法的核函数
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">输入矩阵A</param>
/// <param name="pMatrixB">输入矩阵B</param>
/// <param name="pMatrixC">输出矩阵C</param>
/// <param name="nMatrixShapeM">矩阵C的行数</param>
/// <param name="nMatrixShapeN">矩阵C的列数</param>
/// <param name="nMatrixShapeK">矩阵A的行数</param>
/// <param name="nBlockSize">数据划分的块大小</param>
/// <param name="pstDPCQueue">DPC++的设备队列</param>
/// <returns></returns>
int MatrixMulti_GPU_SLM_Kernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// 矩阵乘法的核函数
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">输入矩阵A</param>
/// <param name="pMatrixB">输入矩阵B</param>
/// <param name="pMatrixC">输出矩阵C</param>
/// <param name="nMatrixShapeM">矩阵C的行数</param>
/// <param name="nMatrixShapeN">矩阵C的列数</param>
/// <param name="nMatrixShapeK">矩阵A的行数</param>
/// <param name="nBlockSize">数据划分的块大小</param>
/// <param name="pstDPCQueue">DPC++的设备队列</param>
/// <returns></returns>
int MatrixMulti_GPU_SLM_SubMatrix_Kernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// 矩阵乘法的核函数
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">输入矩阵A</param>
/// <param name="pMatrixB">输入矩阵B</param>
/// <param name="pMatrixC">输出矩阵C</param>
/// <param name="nMatrixShapeM">矩阵C的行数</param>
/// <param name="nMatrixShapeN">矩阵C的列数</param>
/// <param name="nMatrixShapeK">矩阵A的行数</param>
/// <param name="nBlockSize">数据划分的块大小</param>
/// <param name="pstDPCQueue">DPC++的设备队列</param>
/// <returns></returns>
int MatrixMulti_OMPKernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK);

/// <summary>
/// 矩阵转置
/// </summary>
/// <param name="pMatrixIn"></param>
/// <param name="pMatrixOut"></param>
/// <param name="nMatrixShapeM"></param>
/// <param name="nMatrixShapeN"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int MatrixTranspose_GPUKernel(ValueType* pMatrixIn, ValueType* pMatrixOut, int nMatrixShapeM, int nMatrixShapeN,
	int nBlockSize, sycl::queue* pstDPCQueue);
