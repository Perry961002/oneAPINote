#pragma once

#include "BasicKernel.h"

/// <summary>
/// ����˷��ĺ˺���
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">�������A</param>
/// <param name="pMatrixB">�������B</param>
/// <param name="pMatrixC">�������C</param>
/// <param name="nMatrixShapeM">����C������</param>
/// <param name="nMatrixShapeN">����C������</param>
/// <param name="nMatrixShapeK">����A������</param>
/// <param name="nBlockSize">���ݻ��ֵĿ��С</param>
/// <param name="pstDPCQueue">DPC++���豸����</param>
/// <returns></returns>
int MatrixMulti_GPUKernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// ����˷��ĺ˺���
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">�������A</param>
/// <param name="pMatrixB">�������B</param>
/// <param name="pMatrixC">�������C</param>
/// <param name="nMatrixShapeM">����C������</param>
/// <param name="nMatrixShapeN">����C������</param>
/// <param name="nMatrixShapeK">����A������</param>
/// <param name="nBlockSize">���ݻ��ֵĿ��С</param>
/// <param name="pstDPCQueue">DPC++���豸����</param>
/// <returns></returns>
int MatrixMulti_GPU_SLM_Kernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// ����˷��ĺ˺���
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">�������A</param>
/// <param name="pMatrixB">�������B</param>
/// <param name="pMatrixC">�������C</param>
/// <param name="nMatrixShapeM">����C������</param>
/// <param name="nMatrixShapeN">����C������</param>
/// <param name="nMatrixShapeK">����A������</param>
/// <param name="nBlockSize">���ݻ��ֵĿ��С</param>
/// <param name="pstDPCQueue">DPC++���豸����</param>
/// <returns></returns>
int MatrixMulti_GPU_SLM_SubMatrix_Kernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue);

/// <summary>
/// ����˷��ĺ˺���
/// A[M, K] B[K, N] C[M, N]
/// </summary>
/// <param name="pMatrixA">�������A</param>
/// <param name="pMatrixB">�������B</param>
/// <param name="pMatrixC">�������C</param>
/// <param name="nMatrixShapeM">����C������</param>
/// <param name="nMatrixShapeN">����C������</param>
/// <param name="nMatrixShapeK">����A������</param>
/// <param name="nBlockSize">���ݻ��ֵĿ��С</param>
/// <param name="pstDPCQueue">DPC++���豸����</param>
/// <returns></returns>
int MatrixMulti_OMPKernel(ValueType* pMatrixA, ValueType* pMatrixB, ValueType* pMatrixC,
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK);

/// <summary>
/// ����ת��
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
