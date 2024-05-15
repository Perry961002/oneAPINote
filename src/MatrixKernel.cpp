#include "MatrixKernel.h"
#include <omp.h>

/// <summary>
/// 每个Item可在SLM上分配的默认数组长度
/// </summary>
const int nLocalFloatArrayLen = 16;

/// <summary>
/// 每个Item可在SLM上分配的默认子方阵[nSubMatrixSize, nSubMatrixSize]
/// </summary>
const int nSubMatrixSize = 8;

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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK, ValueType alpha, ValueType beta, int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pstDPCQueue || !pMatrixA || !pMatrixB || !pMatrixC ||
		nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nMatrixShapeK <= 0 || nBlockSize <= 0)
	{
		return -1;
	}

	int nRet = 0;
	ValueType* pMatrixTransposeB = nullptr;

	try
	{
		// 按块大小对矩阵形状进行取整
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixShapeN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// 将B矩阵进行转置 以减少内存墙的影响
		pMatrixTransposeB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixB, pMatrixTransposeB, nMatrixShapeK, nMatrixShapeN, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("矩阵B转置失败");
		}
		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nRowIndex = _Item.get_global_id(0);
						int nColIndex = _Item.get_global_id(1);

						// OR
						/*int nRowIndex = _Item.get_local_id(0) + _Item.get_group(0) * _Item.get_local_range(0);
						int nColIndex = _Item.get_local_id(1) + _Item.get_group(1) * _Item.get_local_range(1);*/

						// 保证取得的二维下标是有效的(因为stGlobalRange是按矫正之后的行列构造的)
						if (nRowIndex >= nMatrixShapeM || nColIndex >= nMatrixShapeN) return;

						int nOffestMatrixRowA = nMatrixShapeK * nRowIndex;
						int nOffestMatrixTransposeRowB = nMatrixShapeK * nColIndex;
						ValueType _Sum = 0.0;
						for (int _nIndex = 0; _nIndex < nMatrixShapeK; ++_nIndex)
						{
							//_Sum += pMatrixA[nOffestMatrixRowA + _nIndex] * pMatrixB[nMatrixShapeN * _nIndex + nColIndex];
							_Sum += pMatrixA[nOffestMatrixRowA + _nIndex] * pMatrixTransposeB[nOffestMatrixTransposeRowB + _nIndex];
						}
						_Sum = _Sum * alpha + beta * pMatrixC[nMatrixShapeN * nRowIndex + nColIndex];
						pMatrixC[nMatrixShapeN * nRowIndex + nColIndex] = _Sum;
					});
			}
		);
		// 等待计算完成
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		nRet= ex.code().value();
	}
	catch (std::exception& ex)
	{
		nRet = -1;
	}


	Free(pstDPCQueue, pMatrixTransposeB);
	return nRet;
}

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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK, ValueType alpha, ValueType beta)
{
	if (pMatrixA == nullptr || pMatrixB == nullptr || pMatrixC == nullptr ||
		nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nMatrixShapeK <= 0)
	{
		return -1;
	}

#pragma omp parallel for
	for (int _RowIndex = 0; _RowIndex < nMatrixShapeM; ++_RowIndex)
	{
#pragma omp parallel for
		for (int _ColIndex = 0; _ColIndex < nMatrixShapeN; ++_ColIndex)
		{
			ValueType _Sum = 0;
			int nOffestMatrixRowA = nMatrixShapeK * _RowIndex;
			for (int _K = 0; _K < nMatrixShapeK; ++_K)
			{
				_Sum += pMatrixA[nOffestMatrixRowA + _K] * pMatrixB[nMatrixShapeN * _K + _ColIndex];
			}
			_Sum = _Sum * alpha + beta * pMatrixC[_RowIndex * nMatrixShapeN + _ColIndex];
			pMatrixC[_RowIndex * nMatrixShapeN + _ColIndex] = _Sum;
		}
	}

	return 0;
}

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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK, ValueType alpha, ValueType beta, int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pstDPCQueue || !pMatrixA || !pMatrixB || !pMatrixC ||
		nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nMatrixShapeK <= 0 || nBlockSize <= 0)
	{
		return -1;
	}

	int nRet = 0;
	ValueType* pMatrixTransposeB = nullptr;

	try
	{
		// 按块大小对矩阵形状进行取整
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixShapeN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// 将B矩阵进行转置 以减少内存墙的影响
		pMatrixTransposeB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixB, pMatrixTransposeB, nMatrixShapeK, nMatrixShapeN, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("矩阵B转置失败");
		}

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nRowIndex = _Item.get_global_id(0);
						int nColIndex = _Item.get_global_id(1);

						// 保证取得的二维下标是有效的(因为stGlobalRange是按矫正之后的行列构造的)
						if (nRowIndex >= nMatrixShapeM || nColIndex >= nMatrixShapeN) return;

						int nOffestMatrixRowA = nMatrixShapeK * nRowIndex;
						int nOffestMatrixTransposeRowB = nMatrixShapeK * nColIndex;
						ValueType _Sum = 0.0;

						ValueType arrMatrixTileA[nLocalFloatArrayLen] = { 0.0 };
						ValueType arrMatrixTileB[nLocalFloatArrayLen] = { 0.0 };
						int nLen = nMatrixShapeK / nLocalFloatArrayLen * nLocalFloatArrayLen;
						for (int nLoop = 0; nLoop < nLen; nLoop += nLocalFloatArrayLen)
						{
							for (int nIndex = 0; nIndex < nLocalFloatArrayLen; ++nIndex)
							{
								int nOffest = nLoop + nIndex;
								// 按行装入A的局部数据
								arrMatrixTileA[nIndex] = pMatrixA[nOffestMatrixRowA + nOffest];
								// 按列装入B的局部数据
								//arrMatrixTileB[nIndex] = pMatrixB[nMatrixShapeN * nOffest + nColIndex];
								arrMatrixTileB[nIndex] = pMatrixTransposeB[nOffestMatrixTransposeRowB + nOffest];
							}

							// 计算_Sum
							for (int _nIndex = 0; _nIndex < nLocalFloatArrayLen; ++_nIndex)
							{
								_Sum += arrMatrixTileA[_nIndex] * arrMatrixTileB[_nIndex];
							}
						}

						for (int nIndex = nLen; nIndex < nMatrixShapeK; ++nIndex)
						{
							_Sum += pMatrixA[nOffestMatrixRowA + nIndex] * pMatrixTransposeB[nOffestMatrixTransposeRowB + nIndex];
						}
						_Sum = _Sum * alpha + beta * pMatrixC[nMatrixShapeN * nRowIndex + nColIndex];
						pMatrixC[nMatrixShapeN * nRowIndex + nColIndex] = _Sum;
					});
			}
		);
		// 等待计算完成
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		nRet = ex.code().value();
	}
	catch (std::exception& ex)
	{
		nRet = -1;
	}

	Free(pstDPCQueue, pMatrixTransposeB);
	return nRet;
}

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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK, ValueType alpha, ValueType beta, int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pstDPCQueue || !pMatrixA || !pMatrixB || !pMatrixC ||
		nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nMatrixShapeK <= 0 || nBlockSize <= 0)
	{
		return -1;
	}

	int nRet = 0;
	ValueType* pMatrixTransposeA = nullptr;
	ValueType* pInnerMatrixA = nullptr;
	ValueType* pInnerMatrixB = nullptr;

	try
	{
		// 按子矩阵大小调整输入矩阵大小
		int nMatrixM = (nMatrixShapeM + nSubMatrixSize - 1) / nSubMatrixSize;
		int nMatrixN = (nMatrixShapeN + nSubMatrixSize - 1) / nSubMatrixSize;
		// 按块大小对矩阵形状进行取整
		int nGridRows = (nMatrixM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// 将A矩阵进行转置 以减少内存墙的影响
		pMatrixTransposeA = Malloc(pstDPCQueue, nMatrixShapeM * nMatrixShapeK, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixA, pMatrixTransposeA, nMatrixShapeM, nMatrixShapeK, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("矩阵A转置失败");
		}

		// 因为后面每个nd_item处理的都是A和B的一个K*nSubMatrixSize大小的子矩阵, 可以改变存储方式来加速
		pInnerMatrixA = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixM * nSubMatrixSize, EMemoryAlloc::Device);
		if (pInnerMatrixA == nullptr) throw std::exception("快速访问矩阵A内存分配失败");
		pInnerMatrixB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixN * nSubMatrixSize, EMemoryAlloc::Device);
		if (pInnerMatrixB == nullptr) throw std::exception("快速访问矩阵B内存分配失败");

		int nTempGlobalLen = std::max(nMatrixM, nMatrixN);
		sycl::range<1> TempRange(nTempGlobalLen);
		auto stPrevTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle) 
			{
				stDPCHandle.parallel_for(TempRange, [=](sycl::item<1> _Item)
					{
						int nGlobalIndex = _Item.get_id(0);
						int nBeginColIndex = nGlobalIndex * nSubMatrixSize;
						int nGlobalOffset = nSubMatrixSize * nMatrixShapeK * nGlobalIndex;
						// 拷贝每一行从nBeginColIndex列开始的nSubMatrixSize个数据
						for (int k = 0; k < nMatrixShapeK; ++k)
						{
							int nOffset = nGlobalOffset + k * nSubMatrixSize;
							// 拷贝A
							for (int i = 0; i < nSubMatrixSize; ++i)
							{
								if (nBeginColIndex + i < nMatrixShapeM)
								{
									pInnerMatrixA[nOffset + i] = pMatrixTransposeA[k * nMatrixShapeM + nBeginColIndex + i];
								}
								else
								{
									pInnerMatrixA[nOffset + i] = 0;
								}
								
							}
						}

						for (int k = 0; k < nMatrixShapeK; ++k)
						{
							int nOffset = nGlobalOffset + k * nSubMatrixSize;
							// 拷贝B
							for (int i = 0; i < nSubMatrixSize; ++i)
							{
								if (nBeginColIndex + i < nMatrixShapeN)
								{
									pInnerMatrixB[nOffset + i] = pMatrixB[k * nMatrixShapeN + nBeginColIndex + i];
								}
								else
								{
									pInnerMatrixB[nOffset + i] = 0;
								}

							}
						}
					});
			});
		stPrevTaskEvent.wait();
		Free(pstDPCQueue, pMatrixTransposeA);
		pMatrixTransposeA = nullptr;

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nItemRow = _Item.get_global_id(0);
						int nItemCol = _Item.get_global_id(1);
						int nGlobalRow = nItemRow * nSubMatrixSize;
						int nGlobalCol = nItemCol * nSubMatrixSize;
						// 保证取得的二维下标是有效的(因为stGlobalRange是按两次取整矫正之后的行列构造的)
						if (nGlobalRow >= nMatrixShapeM || nGlobalCol >= nMatrixShapeN) return;

						ValueType arrTileA[nSubMatrixSize] = { 0 };
						ValueType arrTileB[nSubMatrixSize] = { 0 };
						ValueType arrSubMatrix[nSubMatrixSize][nSubMatrixSize] = { 0 };

						int nGlobalInnerMatricAOffset = nSubMatrixSize * nMatrixShapeK * nItemRow;
						int nGlobalInnerMatricBOffset = nSubMatrixSize * nMatrixShapeK * nItemCol;
						// 矩阵A已经转置为K*M
						for (int k = 0; k < nMatrixShapeK; ++k)
						{
							int nInnerMatrixAOffset = k * nSubMatrixSize;
							for (int m = 0; m < nSubMatrixSize; ++m)
							{
								arrTileA[m] = pInnerMatrixA[nGlobalInnerMatricAOffset + nInnerMatrixAOffset + m];
							}


							int nInnerMatrixBOffset = k * nSubMatrixSize;
							for (int n = 0; n < nSubMatrixSize; ++n)
							{
								arrTileB[n] = pInnerMatrixB[nGlobalInnerMatricBOffset + nInnerMatrixBOffset + n];
							}

							for (int m = 0; m < nSubMatrixSize; ++m)
							{
								for (int n = 0; n < nSubMatrixSize; ++n)
								{
									arrSubMatrix[m][n] += arrTileA[m] * arrTileB[n];
								}
							}
						}

						for (int m = 0; m < nSubMatrixSize; ++m)
						{
							if (nGlobalRow + m >= nMatrixShapeM) break;
							ValueType* pCurrOutMatrixRow = pMatrixC + (nGlobalRow + m) * nMatrixShapeN;
							for (int n = 0; n < nSubMatrixSize; ++n)
							{
								if (nGlobalCol + n >= nMatrixShapeN) break;
								ValueType _Origin = (*(pCurrOutMatrixRow + nGlobalCol + n)) * beta;
								arrSubMatrix[m][n] = alpha * arrSubMatrix[m][n] + _Origin;
							}
						}

						for (int m = 0; m < nSubMatrixSize; ++m)
						{
							if (nGlobalRow + m >= nMatrixShapeM) break;
							ValueType* pCurrOutMatrixRow = pMatrixC + (nGlobalRow + m) * nMatrixShapeN + nGlobalCol;
							ValueType* pSubRow = &(arrSubMatrix[m][0]);
							for (int n = 0; n < nSubMatrixSize; ++n)
							{
								if (nGlobalCol + n >= nMatrixShapeN) break;
								pCurrOutMatrixRow[n] = pSubRow[n];
							}
						}
					});
			}
		);
		// 等待计算完成
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		nRet = ex.code().value();
	}
	catch (std::exception& ex)
	{
		nRet = -1;
	}

	Free(pstDPCQueue, pMatrixTransposeA);
	Free(pstDPCQueue, pInnerMatrixA);
	Free(pstDPCQueue, pInnerMatrixB);
	return nRet;
}

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
	int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pstDPCQueue || !pMatrixIn || !pMatrixOut ||nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nBlockSize <= 0)
	{
		return -1;
	}

	try
	{
		int nSubVectorLen = nLocalFloatArrayLen * 2;
		// 将每一行划分成小块
		int nMatrixN = (nMatrixShapeN + nSubVectorLen - 1) / nSubVectorLen;
		// 按块大小对矩阵形状进行取整
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nRowIndex = _Item.get_global_id(0);
						int nColBeginIndex = _Item.get_global_id(1) * nSubVectorLen;

						// 保证取得的二维下标是有效的(因为stGlobalRange是按矫正之后的行列构造的)
						if (nRowIndex >= nMatrixShapeM || nColBeginIndex >= nMatrixShapeN) return;

						int nInMatrixRowOffset = nRowIndex * nMatrixShapeN;
						// 转置一个子向量
						for (int nColIndex = nColBeginIndex, nOffset = 0; nColIndex < nMatrixShapeN && nOffset < nSubVectorLen;
							++nColIndex, ++nOffset)
						{
							pMatrixOut[nColIndex * nMatrixShapeM + nRowIndex] = pMatrixIn[nInMatrixRowOffset + nColIndex];
						}
					});
			}
		);
		// 等待计算完成
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		return ex.code().value();
	}

	return 0;
}

int MatrixConvolution_GPUKernel(ValueType* pMatrixInput, int nInputM, int nInputN,
	ValueType* pMatrixKernel, int nKernelM, int nKernelN, ValueType* pMatrixOutput,
	int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pMatrixInput || !pMatrixKernel || !pMatrixOutput || !pstDPCQueue ||
		nInputM <= 0 || nInputN <= 0 || nKernelM <= 0 || nKernelN <= 0 ||
		nInputM <= nKernelM || nInputN <= nKernelN || nBlockSize <= 0)
	{
		return -1;
	}

	try
	{
		// 输出矩阵的形状
		int nOutputM = nInputM - nKernelM + 1;
		 int nOutputN = nInputN - nKernelN + 1;
		// 矫正任务的全局范围
		int nGridRows = ((nOutputM + nSubMatrixSize - 1) / nSubMatrixSize + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = ((nOutputN + nSubMatrixSize - 1) / nSubMatrixSize + nBlockSize - 1) / nBlockSize * nBlockSize;
		
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nOutputBeginRow = _Item.get_global_id(0) * nSubMatrixSize;
						int nOutputBeginCol = _Item.get_global_id(1) * nSubMatrixSize;

						for (int m = 0; m < nSubMatrixSize; ++m)
						{
							int nOutRowIndex = nOutputBeginRow + m;
							if (nOutRowIndex >= nOutputM) break;
							int nOutRowOffset = nOutRowIndex * nOutputN;
							for (int n = 0; n < nSubMatrixSize; ++n)
							{
								int nOutColIndex = nOutputBeginCol + n;
								if (nOutColIndex >= nOutputN) break;

								ValueType _Sum = 0;

								for (int nKernelRow = 0; nKernelRow < nKernelM; ++nKernelRow)
								{
									ValueType* pMatrixIn = pMatrixInput + (nOutRowIndex + nKernelRow) * nInputN + nOutColIndex;
									int nKernelRowOffset = nKernelRow * nKernelN;
									for (int nKernelCol = 0; nKernelCol < nKernelN; ++nKernelCol)
									{
										ValueType _In = pMatrixIn[nKernelCol];
										ValueType _Kernel = pMatrixKernel[nKernelRowOffset + nKernelCol];
										_Sum += _In * _Kernel;
									}
								}

								pMatrixOutput[nOutRowOffset + nOutColIndex] = _Sum;
							}
						}

					});
			});
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		return ex.code().value();
	}

	return 0;
}


int MatrixConvolution_GPUKernel_V2(ValueType* pMatrixInput, int nInputM, int nInputN,
	ValueType* pMatrixKernel, int nKernelM, int nKernelN, ValueType* pMatrixOutput,
	int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pMatrixInput || !pMatrixKernel || !pMatrixOutput || !pstDPCQueue ||
		nInputM <= 0 || nInputN <= 0 || nKernelM <= 0 || nKernelN <= 0 ||
		nInputM <= nKernelM || nInputN <= nKernelN || nBlockSize <= 0)
	{
		return -1;
	}

	try
	{
		// 输出矩阵的形状
		int nOutputM = nInputM - nKernelM + 1;
		int nOutputN = nInputN - nKernelN + 1;
		// 矫正任务的全局范围
		int nGridRows = (nOutputM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nOutputN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// 初始化nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nOutputRowIndex = _Item.get_global_id(0);
						int nOutputColIndex = _Item.get_global_id(1);
						// 保证输出矩阵的下标没有越界
						if (nOutputRowIndex >= nOutputM || nOutputColIndex >= nOutputN) return;
						ValueType _Sum = 0;
						ValueType arrInput[nSubMatrixSize] = { 0 };
						ValueType arrKernel[nSubMatrixSize] = { 0 };
						int nUsedPosition = 0;
						for (int m = 0; m < nKernelM; ++m)
						{
							int nInputRowOffset = (nOutputRowIndex + m) * nInputN + nOutputColIndex;
							int nKernelRowOffset = m * nKernelN;
							for (int n = 0; n < nKernelN; ++n)
							{
								arrInput[nUsedPosition] = pMatrixInput[nInputRowOffset + n];
								arrKernel[nUsedPosition] = pMatrixKernel[nKernelRowOffset + n];
								++nUsedPosition;
								if (nUsedPosition == nSubMatrixSize)
								{
									for (int nIndex = 0; nIndex < nSubMatrixSize; ++nIndex)
									{
										_Sum += arrInput[nIndex] * arrKernel[nIndex];
										arrInput[nIndex] = 0; 
										arrKernel[nIndex] = 0;
									}
									nUsedPosition = 0;
								}
							}
						}

						for (int nIndex = 0; nIndex < nSubMatrixSize; ++nIndex)
						{
							_Sum += arrInput[nIndex] * arrKernel[nIndex];
						}
						pMatrixOutput[nOutputRowIndex * nOutputN + nOutputColIndex] = _Sum;
					});
			});
		stTaskEvent.wait();
	}
	catch (sycl::exception& ex)
	{
		std::cout << ex.code().message() << std::endl;
		return ex.code().value();
	}

	return 0;
}

/// <summary>
/// 卷积
/// </summary>
/// <param name="pMatrixInput"></param>
/// <param name="nInputM"></param>
/// <param name="nInputN"></param>
/// <param name="pMatrixKernel"></param>
/// <param name="nKernelM"></param>
/// <param name="nKernelN"></param>
/// <param name="pMatrixOutput">OutputM=nInputM-nKernelM+1, OutputN=nInputN-nKernelN+1</param>
/// <returns></returns>
int MatrixConvolution_CPUKernel(ValueType* pMatrixInput, int nInputM, int nInputN,
	ValueType* pMatrixKernel, int nKernelM, int nKernelN, ValueType* pMatrixOutput)
{
	if (!pMatrixInput || !pMatrixKernel || !pMatrixOutput || nInputM <= 0 || nInputN <= 0 || nKernelM <= 0 || nKernelN <= 0 ||
		nInputM <= nKernelM || nInputN <= nKernelN)
	{
		return -1;
	}

	// 输出矩阵的形状
	int nOutputM = nInputM - nKernelM + 1;
	int nOutputN = nInputN - nKernelN + 1;

	for (int nOutputRowIndex = 0; nOutputRowIndex < nOutputM; ++nOutputRowIndex)
	{
		ValueType* pOutRow = pMatrixOutput + nOutputRowIndex * nOutputN;
		for (int nOutputColIndex = 0; nOutputColIndex < nOutputN; ++nOutputColIndex)
		{
			ValueType _SubSum = 0.0;
			for (int nKernelRowIndex = 0; nKernelRowIndex < nKernelM; ++nKernelRowIndex)
			{
				ValueType* pKernelRow = pMatrixKernel + nKernelRowIndex * nKernelN;
				ValueType* pInputRow = pMatrixInput + (nOutputRowIndex + nKernelRowIndex) * nInputN + nOutputColIndex;
				for (int nKernelColIndex = 0; nKernelColIndex < nKernelN; ++nKernelColIndex)
				{
					_SubSum += pInputRow[nKernelColIndex] * pKernelRow[nKernelColIndex];
				}
			}
			pOutRow[nOutputColIndex] = _SubSum;
		}
	}

	return 0;
}
