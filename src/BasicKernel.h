#pragma once

#include <sycl/sycl.hpp>

using ValueType = float;

enum EMemoryAlloc
{
	Host,
	Device,
	Shared
};

/// <summary>
/// �����������
/// </summary>
/// <returns></returns>
sycl::queue* CreateDPCQueue();

/// <summary>
/// ��ȡ�������
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetMaxWorkItemSizes(sycl::queue* pstDPCQueue);

/// <summary>
/// ��ȡ������ĵı����ڴ��С
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetLoaclMemorySize(sycl::queue* pstDPCQueue);

/// <summary>
/// �����ڴ�
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="nSize"></param>
/// <param name="eMemoryAlloc"></param>
/// <returns></returns>
ValueType* Malloc(sycl::queue* pstDPCQueue, int nSize, EMemoryAlloc eMemoryAlloc = EMemoryAlloc::Host);

/// <summary>
/// �ͷ��ڴ�
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="pData"></param>
void Free(sycl::queue* pstDPCQueue, ValueType* pData);
