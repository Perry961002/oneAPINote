#include "BasicKernel.h"

#ifdef _DEBUG
#include <string>
#include <iostream>
#endif // _DEBUG

/// <summary>
/// 创建计算队列
/// </summary>
/// <returns></returns>
sycl::queue* CreateDPCQueue()
{
	// 根据默认的设备选择器创建队列
	sycl::queue* pstRetQueue = new(std::nothrow) sycl::queue(sycl::default_selector_v);
	if (pstRetQueue == nullptr) return nullptr;

#ifdef _DEBUG
	std::string strDeviceClassName;
	if (pstRetQueue->get_device().is_cpu()) strDeviceClassName.assign("CPU");
	else if (pstRetQueue->get_device().is_gpu()) strDeviceClassName.assign("GPU");
	else strDeviceClassName.assign("其他设备");
	std::cout << "设备类型: " << strDeviceClassName << std::endl
		<< "设备名称: " <<
		pstRetQueue->get_device().get_info<sycl::info::device::name>()
		<< std::endl << "设备内存: " <<
		pstRetQueue->get_device().get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << "GB"
		<< std::endl << "WorkGroup本地内存: " << pstRetQueue->get_device().get_info<sycl::info::device::local_mem_size>() / 1024.0 << "KB"
		<< std::endl << "最大工作组数: "
		<< pstRetQueue->get_device().get_info<sycl::info::device::max_work_group_size>()
		<< std::endl << "工作组下的最大工作项数: "
		<< pstRetQueue->get_device().get_info<sycl::info::device::max_work_item_sizes<1>>().size() << std::endl;
#endif // _DEBUG

	return pstRetQueue;
}

/// <summary>
/// 获取最大工作项
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetMaxWorkItemSizes(sycl::queue* pstDPCQueue)
{
	return (pstDPCQueue != nullptr ? pstDPCQueue->get_device().get_info<sycl::info::device::max_work_item_sizes<1>>().size() : (size_t)0);
}

/// <summary>
/// 获取计算核心的本地内存大小
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <returns></returns>
size_t GetLoaclMemorySize(sycl::queue* pstDPCQueue)
{
	return (pstDPCQueue != nullptr ? pstDPCQueue->get_device().get_info<sycl::info::device::local_mem_size>() : (size_t)0);
}

/// <summary>
/// 分配内存
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="nSize"></param>
/// <param name="eMemoryAlloc"></param>
/// <returns></returns>
ValueType* Malloc(sycl::queue* pstDPCQueue, int nSize, EMemoryAlloc eMemoryAlloc)
{
	if (pstDPCQueue == nullptr || nSize <= 0) return nullptr;

	ValueType* nRet = nullptr;
	try
	{
		switch (eMemoryAlloc)
		{
		case Host:
			nRet = sycl::malloc_host<ValueType>(nSize, *pstDPCQueue);
			break;
		case Device:
			nRet = sycl::malloc_device<ValueType>(nSize, *pstDPCQueue);
			break;
		case Shared:
			nRet = sycl::malloc_shared<ValueType>(nSize, *pstDPCQueue);
			break;
		default:
			break;
		}
	}
	catch (const std::exception&)
	{
		return nullptr;
	}

	return nRet;
}

/// <summary>
/// 释放内存
/// </summary>
/// <param name="pstDPCQueue"></param>
/// <param name="pData"></param>
void Free(sycl::queue* pstDPCQueue, ValueType* pData)
{
	if (pstDPCQueue != nullptr && pData != nullptr)
	{
		sycl::free(pData, *pstDPCQueue);
	}
}
