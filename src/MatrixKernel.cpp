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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue)
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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK)
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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue)
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
	int nMatrixShapeM, int nMatrixShapeN, int nMatrixShapeK,
	int nBlockSize, sycl::queue* pstDPCQueue)
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

								//SubMatrix[m][n] += _Sum;
								pMatrixC[nOffsetMatrixRowC + nColIndex] = _Sum;
							}
						}

						// д�����Ӿ���
						/*for (int m = 0; m < nSubMatrixSize; m++) {
							int nRowIndex = nGlobalRow + m;
							if (nRowIndex >= nMatrixShapeM) break;
							int nOffset = nRowIndex * nMatrixShapeN;
							for (int n = 0; n < nSubMatrixSize; n++) {
								int nColIndex = nGlobalCol + n;
								if (nColIndex >= nMatrixShapeN) break;
								pMatrixC[nOffset + nColIndex] = SubMatrix[m][n];
							}
						}*/
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
