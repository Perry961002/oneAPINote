#pragma once

#include "MatrixKernel.h"
#include <random>
#include <mkl.h>

/// <summary>
/// 初始化矩阵数据
/// A[M, K] B[K, N]
/// </summary>
/// <param name="pMatrixA"></param>
/// <param name="pMatrixB"></param>
/// <param name="nMatrixShapeM"></param>
/// <param name="nMatrixShapeN"></param>
/// <param name="nMatrixShapeK"></param>
void InitMatrixData(ValueType* pMatrixA, ValueType* pMatrixB, int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK)
{
	if (pMatrixA == nullptr || pMatrixB == nullptr ||
		nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nMatrixShapeK <= 0)
	{
		return;
	}
	std::random_device rd;
	std::mt19937 stMt19937{ rd() };
	std::uniform_real_distribution<ValueType> stRealDistri(-10.0, 10.0);

	int nMatrixSize = nMatrixShapeM * nMatrixShapeK;
	int nIndex = 0;
	for (; nIndex < nMatrixSize; ++nIndex)
	{
		pMatrixA[nIndex] = stRealDistri(stMt19937);
	}

	nMatrixSize = nMatrixShapeK * nMatrixShapeN;
	for (nIndex = 0; nIndex < nMatrixSize; ++nIndex)
	{
		pMatrixB[nIndex] = stRealDistri(stMt19937);
	}
}

bool Verify(ValueType* pMatrixA, ValueType* pMatrixB, int nSize)
{
	bool bIsSame = true;
	for (int nIndex = 0; nIndex < nSize; ++nIndex)
	{
		if (abs(pMatrixA[nIndex] - pMatrixB[nIndex]) > 0.5)
		{
			std::cout << pMatrixA[nIndex] << "\t" << pMatrixB[nIndex] << std::endl;
			bIsSame = false;
			break;
		}
	}
	return bIsSame;
}

void AdjustStringWidth(std::vector<std::string>& vecStrings, bool bIsAppend=true)
{
	if (vecStrings.empty()) return;
	int nMaxLen=std::max_element(vecStrings.begin(), vecStrings.end(), 
		[](std::string& left, std::string& right) {return left.size() < right.size(); })->size();
	for (auto& currStr : vecStrings)
	{
		if (bIsAppend)
		{
			while (currStr.size() < nMaxLen) currStr.append(" ");
		}
		else
		{
			while (currStr.size() < nMaxLen) currStr.insert(0, " ");
		}
	}
}

int TestMatrixKernel()
{
	int nMatrixShapeM = 8192;
	int nMatrixShapeN = 8192;
	int nMatrixShapeK = 3000;
	bool bIsSame = false;

	sycl::queue* pstDPCQueue = CreateDPCQueue();
	if (pstDPCQueue == nullptr)
	{
		std::cout << "设备队列创建失败!" << std::endl;
		return 0;
	}
	std::cout << "=========================================" << std::endl;
	char strMatrixShapeInfo[128];
	sprintf_s(strMatrixShapeInfo, sizeof(strMatrixShapeInfo), "A[%d, %d] * B[%d, %d] = C[%d, %d]",
		nMatrixShapeM, nMatrixShapeK, nMatrixShapeK, nMatrixShapeN, nMatrixShapeM, nMatrixShapeN);
	std::cout << strMatrixShapeInfo << std::endl << "=========================================" << std::endl;

	ValueType* pMatrixA = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeK, EMemoryAlloc::Shared);
	ValueType* pMatrixB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixC = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixD = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixE = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixF = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Host);
	InitMatrixData(pMatrixA, pMatrixB, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK);
	
	int nBlockSize = sqrt((double)GetMaxWorkItemSizes(pstDPCQueue));

	auto _Start = std::chrono::high_resolution_clock::now();
	int nGPURet = MatrixMulti_GPUKernel(pMatrixA, pMatrixB, pMatrixC, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	auto _End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPURet != 0)
	{
		std::cout << "GPU计算失败!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPUSLMRet = MatrixMulti_GPU_SLM_Kernel(pMatrixA, pMatrixB, pMatrixD, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU_SLM = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPUSLMRet != 0)
	{
		std::cout << "GPU_SLM计算失败!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPUSLMSubRet = MatrixMulti_GPU_SLM_SubMatrix_Kernel(pMatrixA, pMatrixB, pMatrixE, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU_SLMSub = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPUSLMSubRet != 0)
	{
		std::cout << "GPU_SLM子矩阵划分计算失败!" << std::endl;
	}

	int nMKLRet = 0;
	_Start = std::chrono::high_resolution_clock::now();
	cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
		nMatrixShapeM, nMatrixShapeN, nMatrixShapeK, 1.0f, pMatrixA, nMatrixShapeK, pMatrixB, nMatrixShapeN, 0.0f, pMatrixF, nMatrixShapeN);
	
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeMKL = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nMKLRet != 0)
	{
		std::cout << "OpenMP计算失败!" << std::endl;
	}

	if (nGPURet == 0 && nMKLRet == 0 && nGPUSLMRet == 0 && nGPUSLMSubRet == 0)
	{
		bIsSame = Verify(pMatrixC, pMatrixD, nMatrixShapeM * nMatrixShapeN) && Verify(pMatrixC, pMatrixE, nMatrixShapeM * nMatrixShapeN) &&
			Verify(pMatrixC, pMatrixF, nMatrixShapeM * nMatrixShapeN);
		if (!bIsSame) std::cout << "计算结果不一致!" << std::endl;
		else
		{
			std::vector<std::string> vecTestNames{ "GPU用时: ","GPU局部缓存加速用时: ",
												"GPU局部缓存&&子矩阵划分加速用时: ","使用Intel MKL用时: " };
			AdjustStringWidth(vecTestNames, true);

			std::vector<std::string> vecTestTimes{ std::to_string(llUseTimeGPU),std::to_string(llUseTimeGPU_SLM),
												std::to_string(llUseTimeGPU_SLMSub),std::to_string(llUseTimeMKL) };
			AdjustStringWidth(vecTestTimes, false);

			std::cout << "计算结果一致" << std::endl;
			for (int nIndex = 0; nIndex < vecTestNames.size(); ++nIndex)
			{
				std::cout << vecTestNames[nIndex] << vecTestTimes[nIndex] << " 微秒" << std::endl;
			}
		}
	}
	
	if (pstDPCQueue != nullptr)
	{
		Free(pstDPCQueue, pMatrixA);
		Free(pstDPCQueue, pMatrixB);
		Free(pstDPCQueue, pMatrixC);
		Free(pstDPCQueue, pMatrixD);
		Free(pstDPCQueue, pMatrixE);
		delete pstDPCQueue;
		pstDPCQueue = nullptr;
	}

	return 0;
}
