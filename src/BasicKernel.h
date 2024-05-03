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
/// 创建计算队列
/// </summary>
/// <returns></returns>
sycl::queue* CreateDPCQueue();

/// <summary>
/// 获取最大工作项
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetMaxWorkItemSizes(sycl::queue* pstDPCQueue);

/// <summary>
/// 获取计算核心的本地内存大小
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetLoaclMemorySize(sycl::queue* pstDPCQueue);

/// <summary>
/// 分配内存
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="nSize"></param>
/// <param name="eMemoryAlloc"></param>
/// <returns></returns>
ValueType* Malloc(sycl::queue* pstDPCQueue, int nSize, EMemoryAlloc eMemoryAlloc = EMemoryAlloc::Host);

/// <summary>
/// 释放内存
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="pData"></param>
void Free(sycl::queue* pstDPCQueue, ValueType* pData);
