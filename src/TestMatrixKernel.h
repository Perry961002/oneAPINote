#pragma once

#include "MatrixKernel.h"
#include <random>
#include <mkl.h>

/// <summary>
/// ��ʼ����������
/// A[M, K] B[K, N]
/// </summary>
/// <param name="pMatrixA"></param>
/// <param name="pMatrixB"></param>
/// <param name="nMatrixShapeM"></param>
/// <param name="nMatrixShapeN"></param>
/// <param name="nMatrixShapeK"></param>
void InitMatrixData(ValueType* pMatrix, int nMatrixShapeM, int nMatrixShapeN)
{
	if (pMatrix == nullptr || nMatrixShapeM <= 0 || nMatrixShapeN <= 0)
	{
		return;
	}
	std::random_device rd;
	std::mt19937 stMt19937{ rd() };
	std::uniform_real_distribution<ValueType> stRealDistri(-10.0, 10.0);

	int nMatrixSize = nMatrixShapeM * nMatrixShapeN;
	int nIndex = 0;
	for (; nIndex < nMatrixSize; ++nIndex)
	{
		pMatrix[nIndex] = stRealDistri(stMt19937);
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
	int nMatrixShapeK = 5000;
	bool bIsSame = false;

	auto _Start = std::chrono::high_resolution_clock::now();
	auto _End = std::chrono::high_resolution_clock::now();

	sycl::queue* pstDPCQueue = CreateDPCQueue();
	if (pstDPCQueue == nullptr)
	{
		std::cout << "�豸���д���ʧ��!" << std::endl;
		return 0;
	}
	std::cout << "=========================================" << std::endl;
	char strMatrixShapeInfo[128];
	sprintf_s(strMatrixShapeInfo, sizeof(strMatrixShapeInfo), "A[%d, %d] * B[%d, %d] = C[%d, %d]",
		nMatrixShapeM, nMatrixShapeK, nMatrixShapeK, nMatrixShapeN, nMatrixShapeM, nMatrixShapeN);
	std::cout << strMatrixShapeInfo << std::endl << "=========================================" << std::endl;

	int nBlockSize = sqrt((double)GetMaxWorkItemSizes(pstDPCQueue));

	ValueType* pMatrixA = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeK, EMemoryAlloc::Shared);
	ValueType* pMatrixB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixC = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixD = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixE = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Shared);
	ValueType* pMatrixF = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeN, EMemoryAlloc::Host);

	InitMatrixData(pMatrixA, nMatrixShapeM, nMatrixShapeK);
	InitMatrixData(pMatrixB, nMatrixShapeK, nMatrixShapeN);

	{
		// ��һ�ε��õĽ����׼ȷ������
		MatrixMulti_GPUKernel(pMatrixA, pMatrixB, pMatrixC, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
			nBlockSize, pstDPCQueue);
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPURet = MatrixMulti_GPUKernel(pMatrixA, pMatrixB, pMatrixC, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPURet != 0)
	{
		std::cout << "GPU����ʧ��!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPUSLMRet = MatrixMulti_GPU_SLM_Kernel(pMatrixA, pMatrixB, pMatrixD, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU_SLM = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPUSLMRet != 0)
	{
		std::cout << "GPU_SLM����ʧ��!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPUSLMSubRet = MatrixMulti_GPU_SLM_SubMatrix_Kernel(pMatrixA, pMatrixB, pMatrixE, nMatrixShapeM, nMatrixShapeN, nMatrixShapeK,
		nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU_SLMSub = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPUSLMSubRet != 0)
	{
		std::cout << "GPU_SLM�Ӿ��󻮷ּ���ʧ��!" << std::endl;
	}

	int nMKLRet = 0;
	_Start = std::chrono::high_resolution_clock::now();
	cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
		nMatrixShapeM, nMatrixShapeN, nMatrixShapeK, 1.0f, pMatrixA, nMatrixShapeK, pMatrixB, nMatrixShapeN, 0.0f, pMatrixF, nMatrixShapeN);

	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeMKL = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nMKLRet != 0)
	{
		std::cout << "OpenMP����ʧ��!" << std::endl;
	}

	if (nGPURet == 0 && nMKLRet == 0 && nGPUSLMRet == 0 && nGPUSLMSubRet == 0)
	{
		bIsSame = Verify(pMatrixC, pMatrixD, nMatrixShapeM * nMatrixShapeN) && Verify(pMatrixC, pMatrixE, nMatrixShapeM * nMatrixShapeN) &&
			Verify(pMatrixC, pMatrixF, nMatrixShapeM * nMatrixShapeN);
		if (!bIsSame) std::cout << "��������һ��!" << std::endl;
		else
		{
			std::vector<std::string> vecTestNames{ "GPU��ʱ: ","GPU�ֲ����������ʱ: ",
												"GPU�ֲ�����&&�Ӿ��󻮷ּ�����ʱ: ","ʹ��Intel MKL��ʱ: " };
			AdjustStringWidth(vecTestNames, true);

			std::vector<std::string> vecTestTimes{ std::to_string(llUseTimeGPU),std::to_string(llUseTimeGPU_SLM),
												std::to_string(llUseTimeGPU_SLMSub),std::to_string(llUseTimeMKL) };
			AdjustStringWidth(vecTestTimes, false);

			std::cout << "������һ��" << std::endl;
			for (int nIndex = 0; nIndex < vecTestNames.size(); ++nIndex)
			{
				std::cout << vecTestNames[nIndex] << vecTestTimes[nIndex] << " ΢��" << std::endl;
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

int TestMatrixConvolution()
{
	int nInputM = 5000;
	int nInputN = 5000;
	int nKernelM = 3;
	int nKernelN = 3;
	int nOutM = nInputM - nKernelM + 1;
	int nOutN = nInputN - nKernelN + 1;

	sycl::queue* pstDPCQueue = CreateDPCQueue();
	if (pstDPCQueue == nullptr)
	{
		std::cout << "�豸���д���ʧ��!" << std::endl;
		return 0;
	}
	std::cout << "=========================================" << std::endl;
	char strMatrixShapeInfo[128];
	sprintf_s(strMatrixShapeInfo, sizeof(strMatrixShapeInfo), "�������[%d, %d], �˺���[%d, %d]", nInputM, nInputN, nKernelM, nKernelN);
	std::cout << strMatrixShapeInfo << std::endl << "=========================================" << std::endl;

	int nBlockSize = sqrt((double)GetMaxWorkItemSizes(pstDPCQueue));

	ValueType* pMatrixIn = Malloc(pstDPCQueue, nInputM * nInputN, EMemoryAlloc::Shared);
	ValueType* pMatrixKernel = Malloc(pstDPCQueue, nKernelM * nKernelN, EMemoryAlloc::Shared);
	ValueType* pMatrixOutA = Malloc(pstDPCQueue, nOutM * nOutN, EMemoryAlloc::Shared);
	ValueType* pMatrixOutB = Malloc(pstDPCQueue, nOutM * nOutN, EMemoryAlloc::Shared);
	ValueType* pMatrixOutC = Malloc(pstDPCQueue, nOutM * nOutN, EMemoryAlloc::Shared);
	std::fill(pMatrixKernel, pMatrixKernel + nKernelM * nKernelN, 1.0);

	{
		// ��һ�ε��ò����Ľ����׼ȷ, ����
		MatrixConvolution_GPUKernel(pMatrixIn, nInputM, nInputN, pMatrixKernel, nKernelM, nKernelN,
			pMatrixOutA, nBlockSize, pstDPCQueue);
	}

	InitMatrixData(pMatrixIn, nInputM, nInputN);

	auto _Start = std::chrono::high_resolution_clock::now();
	int nGPURet = MatrixConvolution_GPUKernel(pMatrixIn, nInputM, nInputN, pMatrixKernel, nKernelM, nKernelN,
		pMatrixOutA, nBlockSize, pstDPCQueue);
	auto _End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPURet != 0)
	{
		std::cout << "GPU����ʧ��!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nGPUV2Ret = MatrixConvolution_GPUKernel_V2(pMatrixIn, nInputM, nInputN, pMatrixKernel, nKernelM, nKernelN,
		pMatrixOutC, nBlockSize, pstDPCQueue);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPUV2 = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPUV2Ret != 0)
	{
		std::cout << "GPUV2����ʧ��!" << std::endl;
	}

	_Start = std::chrono::high_resolution_clock::now();
	int nCPURet = MatrixConvolution_CPUKernel(pMatrixIn, nInputM, nInputN, pMatrixKernel, nKernelM, nKernelN, pMatrixOutB);
	_End = std::chrono::high_resolution_clock::now();
	long long llUseTimeCPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nCPURet != 0)
	{
		std::cout << "CPU����ʧ��!" << std::endl;
	}

	if (nGPURet == 0 && nCPURet == 0&&nGPUV2Ret==0)
	{
		bool bIsSame = Verify(pMatrixOutA, pMatrixOutB, nOutM * nOutN) && Verify(pMatrixOutA, pMatrixOutC, nOutM * nOutN);
		if (!bIsSame) std::cout << "��������һ��!" << std::endl;
		else
		{
			std::vector<std::string> vecTestNames{ "GPU��ʱ: ","GPU���������ʱ: ", "ʹ��CPU��ʱ: " };
			AdjustStringWidth(vecTestNames, true);

			std::vector<std::string> vecTestTimes{ std::to_string(llUseTimeGPU),std::to_string(llUseTimeGPUV2),std::to_string(llUseTimeCPU) };
			AdjustStringWidth(vecTestTimes, false);

			std::cout << "������һ��" << std::endl;
			for (int nIndex = 0; nIndex < vecTestNames.size(); ++nIndex)
			{
				std::cout << vecTestNames[nIndex] << vecTestTimes[nIndex] << " ΢��" << std::endl;
			}
		}
	}

	if (pstDPCQueue != nullptr)
	{
		Free(pstDPCQueue, pMatrixIn);
		Free(pstDPCQueue, pMatrixKernel);
		Free(pstDPCQueue, pMatrixOutA);
		Free(pstDPCQueue, pMatrixOutB);
		Free(pstDPCQueue, pMatrixOutC);

		delete pstDPCQueue;
		pstDPCQueue = nullptr;
	}

	return 0;
}
