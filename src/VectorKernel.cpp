#include "VectorKernel.h"
#include <sycl/sycl.hpp>
#include <immintrin.h>

const int nLocalSubVectorLen = 16;

/// <summary>
/// 求向量方差 GPU
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_GPUKernel(ValueType* pVector, int nVectorSize, int nBlockSize, sycl::queue* pstDPCQueue, ValueType& _OutResult)
{
	if (!pstDPCQueue || !pVector || nVectorSize <= 0 || nBlockSize <= 0) return -1;

	int nRet = 0;
	// 设备内存上开辟的中间结果缓存
	ValueType* pPowSumCache = nullptr;
	ValueType* pSumCache = nullptr;
	try
	{
		// 按局部缓存设定进行分组
		int nSubVectorCount = (nVectorSize + nLocalSubVectorLen - 1) / nLocalSubVectorLen;
		nBlockSize = std::min(nBlockSize, nSubVectorCount);
		// 按BlockSize得到全局范围
		int nGridSize = (nSubVectorCount + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<1> stGlobalRange(nGridSize);
		sycl::range<1> stLocalRange(nBlockSize);
		sycl::nd_range<1> stTaskRange(stGlobalRange, stLocalRange);

		pPowSumCache = Malloc(pstDPCQueue, nGridSize, EMemoryAlloc::Shared);
		if (pPowSumCache == nullptr) return -1;
		memset(pPowSumCache, 0, nGridSize * sizeof(ValueType));

		pSumCache = Malloc(pstDPCQueue, nGridSize, EMemoryAlloc::Shared);
		if (pSumCache == nullptr) return -1;
		memset(pSumCache, 0, nGridSize * sizeof(ValueType));

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<1> _Item)
					{
						int nGlobalIndex = _Item.get_global_id(0);
						int nVectorBlockIndex = nGlobalIndex * nLocalSubVectorLen;
						if (nVectorBlockIndex >= nVectorSize) return;
						
						ValueType arrSubVector[nLocalSubVectorLen] = { 0.0 };
						// 将数组内容拷贝到SLM
						for (int nSubIndex = 0, nVectorIndex = nVectorBlockIndex;
							nSubIndex < nLocalSubVectorLen && nVectorIndex < nVectorSize; ++nSubIndex, ++nVectorIndex)
						{
							arrSubVector[nSubIndex] = pVector[nVectorIndex];
						}

						ValueType _PowSum = 0.0, _Sum = 0.0;
						for (int nIndex = 0; nIndex < nLocalSubVectorLen; ++nIndex)
						{
							ValueType _Curr = arrSubVector[nIndex];
							_Sum += _Curr;
							_PowSum += _Curr * _Curr;
						}

						pPowSumCache[nGlobalIndex] = _PowSum;
						pSumCache[nGlobalIndex] = _Sum;
					});
			}
		);
		// 等待计算完成
		stTaskEvent.wait();
		// 计算最后的方差
		ValueType _PowSum = std::accumulate(pPowSumCache, pPowSumCache + nGridSize, ValueType(0));
		ValueType _Sum = std::accumulate(pSumCache, pSumCache + nGridSize, ValueType(0));
		_OutResult = (_PowSum - (_Sum * _Sum) / nVectorSize) / nVectorSize;
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		nRet = ex.code().value();
	}
	Free(pstDPCQueue, pPowSumCache);
	Free(pstDPCQueue, pSumCache);

	return nRet;
}

/// <summary>
/// 求向量方差 CPU
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_CPUKernel(ValueType* pVector, int nVectorSize, sycl::queue* pstDPCQueue, ValueType& _OutResult)
{
	if (!pstDPCQueue || !pVector || nVectorSize <= 0) return -1;

	// 求平均值
	ValueType _AVG = 0.0;
	for (int nIndex = 0; nIndex < nVectorSize; ++nIndex)
	{
		_AVG += pVector[nIndex];
	}
	_AVG /= nVectorSize;

	_OutResult = 0.0;
	for (int nIndex = 0; nIndex < nVectorSize; ++nIndex)
	{
		ValueType _Diff = pVector[nIndex] - _AVG;
		_OutResult += _Diff * _Diff;
	}
	_OutResult /= nVectorSize;
	return 0;
}

/// <summary>
/// 求向量方差 AVX256
/// </summary>
/// <param name="pVector"></param>
/// <param name="nVectorSize"></param>
/// <param name="nBlockSize"></param>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
int VectorVariance_AVXKernel(ValueType* pVector, int nVectorSize, int nBlockSize, sycl::queue* pstDPCQueue)
{
	return 0;
}
