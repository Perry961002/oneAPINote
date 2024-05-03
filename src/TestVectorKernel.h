#pragma once

#include "VectorKernel.h"
#include <random>

void InitVetorData(ValueType* pVector, int nSize)
{
	if (pVector == nullptr || nSize <= 0) return;
	std::random_device rd;
	std::mt19937 stMt19937{ rd() };
	std::uniform_real_distribution<ValueType> stRealDistri(0.0, 10.0);

	for (int nIndex = 0; nIndex < nSize; ++nIndex)
	{
		pVector[nIndex] = stRealDistri(stMt19937);
	}
}

int TestVectorKernel()
{
	int nVectorSize = 5000000;
	bool bIsSame = false;

	sycl::queue* pstDPCQueue = CreateDPCQueue();
	if (pstDPCQueue == nullptr)
	{
		std::cout << "�豸���д���ʧ��!" << std::endl;
		return 0;
	}
	std::cout << "=========================================" << std::endl;
	std::cout << "��������: " << nVectorSize << std::endl;
	std::cout << "=========================================" << std::endl;

	ValueType* pVectorIn = Malloc(pstDPCQueue, nVectorSize, EMemoryAlloc::Shared);
	InitVetorData(pVectorIn, nVectorSize);

	int nBlockSize = GetMaxWorkItemSizes(pstDPCQueue);

	auto _Start = std::chrono::high_resolution_clock::now();
	ValueType _ResultGPU = 0.0;
	int nGPURet = VectorVariance_GPUKernel(pVectorIn, nVectorSize,nBlockSize, pstDPCQueue, _ResultGPU);
	auto _End = std::chrono::high_resolution_clock::now();
	long long llUseTimeGPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nGPURet != 0)
	{
		std::cout << "GPU����ʧ��!" << std::endl;
	}
	else
	{
		std::cout << "GPU���㷽����: " << _ResultGPU << "\t��ʱ: " << llUseTimeGPU << " ΢��" << std::endl;
	}

	 _Start = std::chrono::high_resolution_clock::now();
	ValueType _ResultCPU = 0.0;
	int nCPURet = VectorVariance_CPUKernel(pVectorIn, nVectorSize, pstDPCQueue, _ResultCPU);
	 _End = std::chrono::high_resolution_clock::now();
	long long llUseTimeCPU = std::chrono::duration_cast<std::chrono::microseconds>(_End - _Start).count();
	if (nCPURet != 0)
	{
		std::cout << "CPU����ʧ��!" << std::endl;
	}
	else
	{
		std::cout << "CPU���㷽����: " << _ResultCPU << "\t��ʱ: " << llUseTimeCPU << " ΢��" << std::endl;
	}
	
	if (pstDPCQueue != nullptr)
	{
		Free(pstDPCQueue, pVectorIn);
		delete pstDPCQueue;
		pstDPCQueue = nullptr;
	}

	return 0;
}

