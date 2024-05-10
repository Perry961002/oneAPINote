#include "MatrixKernel.h"
#include <omp.h>

/// <summary>
/// ÿ��Item����SLM�Ϸ����Ĭ�����鳤��
/// </summary>
const int nLocalFloatArrayLen = 16;

/// <summary>
/// ÿ��Item����SLM�Ϸ����Ĭ���ӷ���[nSubMatrixSize, nSubMatrixSize]
/// </summary>
const int nSubMatrixSize = 8;

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
		// �����С�Ծ�����״����ȡ��
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixShapeN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// ��ʼ��nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// ��B�������ת�� �Լ����ڴ�ǽ��Ӱ��
		pMatrixTransposeB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixB, pMatrixTransposeB, nMatrixShapeK, nMatrixShapeN, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("����Bת��ʧ��");
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

						// ��֤ȡ�õĶ�ά�±�����Ч��(��ΪstGlobalRange�ǰ�����֮������й����)
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
		// �ȴ��������
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
		// �����С�Ծ�����״����ȡ��
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixShapeN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// ��ʼ��nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// ��B�������ת�� �Լ����ڴ�ǽ��Ӱ��
		pMatrixTransposeB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixB, pMatrixTransposeB, nMatrixShapeK, nMatrixShapeN, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("����Bת��ʧ��");
		}

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nRowIndex = _Item.get_global_id(0);
						int nColIndex = _Item.get_global_id(1);

						// ��֤ȡ�õĶ�ά�±�����Ч��(��ΪstGlobalRange�ǰ�����֮������й����)
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
								// ����װ��A�ľֲ�����
								arrMatrixTileA[nIndex] = pMatrixA[nOffestMatrixRowA + nOffest];
								// ����װ��B�ľֲ�����
								//arrMatrixTileB[nIndex] = pMatrixB[nMatrixShapeN * nOffest + nColIndex];
								arrMatrixTileB[nIndex] = pMatrixTransposeB[nOffestMatrixTransposeRowB + nOffest];
							}

							// ����_Sum
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
		// �ȴ��������
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
		// ���Ӿ����С������������С
		int nMatrixM = (nMatrixShapeM + nSubMatrixSize - 1) / nSubMatrixSize;
		int nMatrixN = (nMatrixShapeN + nSubMatrixSize - 1) / nSubMatrixSize;
		// �����С�Ծ�����״����ȡ��
		int nGridRows = (nMatrixM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// ��ʼ��nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		// ��B�������ת�� �Լ����ڴ�ǽ��Ӱ��
		pMatrixTransposeB = Malloc(pstDPCQueue, nMatrixShapeK * nMatrixShapeN, EMemoryAlloc::Device);
		int nTransposeRnt = MatrixTranspose_GPUKernel(pMatrixB, pMatrixTransposeB, nMatrixShapeK, nMatrixShapeN, nBlockSize, pstDPCQueue);
		if (nTransposeRnt != 0)
		{
			throw std::exception("����Bת��ʧ��");
		}

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nGlobalRow = _Item.get_global_id(0) * nSubMatrixSize;
						int nGlobalCol = _Item.get_global_id(1) * nSubMatrixSize;
						// ��֤ȡ�õĶ�ά�±�����Ч��(��ΪstGlobalRange�ǰ�����ȡ������֮������й����)
						if (nGlobalRow >= nMatrixShapeM || nGlobalCol >= nMatrixShapeN) return;

						// ÿ��nd_item��Ӧ������������һ��nSubMatrixSize*nSubMatrixSize�Ӿ���
						// ��Ӧ��������A����nSubMatrixSize*nMatrixShapeK���Ӿ��󣻶�Ӧ��������B����nMatrixShapeK*nSubMatrixSize���Ӿ���
						ValueType arrMatrixTileA[nSubMatrixSize] = { 0.0 };
						ValueType arrMatrixTileB[nSubMatrixSize] = { 0.0 };
						//ValueType SubMatrix[nSubMatrixSize][nSubMatrixSize] = { 0.0 };

						// �����Ӿ���
						for (int m = 0; m < nSubMatrixSize; m++)
						{
							int nRowIndex = nGlobalRow + m;
							if (nRowIndex >= nMatrixShapeM) break;
							int nOffsetMatrixRowC = nRowIndex * nMatrixShapeN;
							int nOffestMatrixRowA = nMatrixShapeK * nRowIndex;
							for (int n = 0; n < nSubMatrixSize; n++) 
							{
								int nColIndex = nGlobalCol + n;
								if (nColIndex >= nMatrixShapeN) break;
								int nOffestMatrixTransposeRowB = nMatrixShapeK * nColIndex;
								ValueType _Sum = 0;
								int nLen = nMatrixShapeK / nLocalFloatArrayLen * nLocalFloatArrayLen;
								for (int nLoop = 0; nLoop < nLen; nLoop += nLocalFloatArrayLen)
								{
									for (int nIndex = 0; nIndex < nLocalFloatArrayLen; ++nIndex)
									{
										int nOffest = nLoop + nIndex;
										// ����װ��A�ľֲ�����
										arrMatrixTileA[nIndex] = pMatrixA[nOffestMatrixRowA + nOffest];
										// ����װ��B�ľֲ�����
										//arrMatrixTileB[nIndex] = pMatrixB[nMatrixShapeN * nOffest + nColIndex];
										arrMatrixTileB[nIndex] = pMatrixTransposeB[nOffestMatrixTransposeRowB + nOffest];
									}

									// ����_Sum
									for (int _nIndex = 0; _nIndex < nLocalFloatArrayLen; ++_nIndex)
									{
										_Sum += arrMatrixTileA[_nIndex] * arrMatrixTileB[_nIndex];
									}
								}

								for (int nIndex = nLen; nIndex < nMatrixShapeK; ++nIndex)
								{
									_Sum += pMatrixA[nOffestMatrixRowA + nIndex] * pMatrixTransposeB[nOffestMatrixTransposeRowB + nIndex];
								}

								_Sum = _Sum * alpha * beta * pMatrixC[nOffsetMatrixRowC + nColIndex];
								pMatrixC[nOffsetMatrixRowC + nColIndex] = _Sum;
							}
						}
					});
			}
		);
		// �ȴ��������
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
	int nBlockSize, sycl::queue* pstDPCQueue)
{
	if (!pstDPCQueue || !pMatrixIn || !pMatrixOut ||nMatrixShapeM <= 0 || nMatrixShapeN <= 0 || nBlockSize <= 0)
	{
		return -1;
	}

	try
	{
		int nSubVectorLen = nLocalFloatArrayLen * 2;
		// ��ÿһ�л��ֳ�С��
		int nMatrixN = (nMatrixShapeN + nSubVectorLen - 1) / nSubVectorLen;
		// �����С�Ծ�����״����ȡ��
		int nGridRows = (nMatrixShapeM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nMatrixN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// ��ʼ��nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nRowIndex = _Item.get_global_id(0);
						int nColBeginIndex = _Item.get_global_id(1) * nSubVectorLen;

						// ��֤ȡ�õĶ�ά�±�����Ч��(��ΪstGlobalRange�ǰ�����֮������й����)
						if (nRowIndex >= nMatrixShapeM || nColBeginIndex >= nMatrixShapeN) return;

						int nInMatrixRowOffset = nRowIndex * nMatrixShapeN;
						// ת��һ��������
						for (int nColIndex = nColBeginIndex, nOffset = 0; nColIndex < nMatrixShapeN && nOffset < nSubVectorLen;
							++nColIndex, ++nOffset)
						{
							pMatrixOut[nColIndex * nMatrixShapeM + nRowIndex] = pMatrixIn[nInMatrixRowOffset + nColIndex];
						}
					});
			}
		);
		// �ȴ��������
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
		// ����������״
		int nOutputM = nInputM - nKernelM + 1;
		 int nOutputN = nInputN - nKernelN + 1;
		// ���������ȫ�ַ�Χ
		int nGridRows = ((nOutputM + nSubMatrixSize - 1) / nSubMatrixSize + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = ((nOutputN + nSubMatrixSize - 1) / nSubMatrixSize + nBlockSize - 1) / nBlockSize * nBlockSize;
		
		// ��ʼ��nd_range
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
		// ����������״
		int nOutputM = nInputM - nKernelM + 1;
		int nOutputN = nInputN - nKernelN + 1;
		// ���������ȫ�ַ�Χ
		int nGridRows = (nOutputM + nBlockSize - 1) / nBlockSize * nBlockSize;
		int nGridCols = (nOutputN + nBlockSize - 1) / nBlockSize * nBlockSize;
		// ��ʼ��nd_range
		sycl::range<2> stGlobalRange(nGridRows, nGridCols);
		sycl::range<2> stLocalRange(nBlockSize, nBlockSize);
		sycl::nd_range<2> stTaskRange(stGlobalRange, stLocalRange);

		auto stTaskEvent = pstDPCQueue->submit([&](sycl::handler& stDPCHandle)
			{
				stDPCHandle.parallel_for(stTaskRange, [=](sycl::nd_item<2> _Item)
					{
						int nOutputRowIndex = _Item.get_global_id(0);
						int nOutputColIndex = _Item.get_global_id(1);
						// ��֤���������±�û��Խ��
						if (nOutputRowIndex >= nOutputM || nOutputColIndex >= nOutputN) return;
						ValueType _Sum = 0;
						ValueType arrInput[nLocalFloatArrayLen] = { 0 };
						ValueType arrKernel[nLocalFloatArrayLen] = { 0 };
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
								if (nUsedPosition == nLocalFloatArrayLen)
								{
									for (int nIndex = 0; nIndex < nLocalFloatArrayLen; ++nIndex)
									{
										_Sum += arrInput[nIndex] * arrKernel[nIndex];
										arrInput[nIndex] = 0; 
										arrKernel[nIndex] = 0;
									}
									nUsedPosition = 0;
								}
							}
						}

						for (int nIndex = 0; nIndex < nLocalFloatArrayLen; ++nIndex)
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
/// ����
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

	// ����������״
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