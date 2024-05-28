# oneAPI学习笔记
- 此项目记录的是[`Intel oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)的学习和测试代码，我的机器使用的CPU是`Intel Ultra 5 125H`，它是一款集成了GPU的SOC。

  下面是利用[`clpeak`](https://github.com/krrishnarraj/clpeak)对GPU进行测试的结果（其他更详细的结果可以[打开这个文件](DeviceBenchmarkResult.txt)）

  ```
  Platform: Intel(R) OpenCL Graphics
    Device: Intel(R) Arc(TM) Graphics
      Driver version  : 31.0.101.5333 (Win64)
      Compute units   : 112
      Clock frequency : 2200 MHz
  
      Global memory bandwidth (GBPS)
        float   : 89.53
        float2  : 81.88
        float4  : 83.05
        float8  : 89.89
        float16 : 86.35
  
      Single-precision compute (GFLOPS)
        float   : 3908.99
        float2  : 3880.27
        float4  : 3888.38
        float8  : 3873.17
        float16 : 3672.71
  
      Double-precision compute (GFLOPS)
        double   : 121.60
        double2  : 120.05
        double4  : 121.75
        double8  : 120.91
        double16 : 116.89
  
      Integer compute (GIOPS)
        int   : 1274.43
        int2  : 1261.42
        int4  : 1255.07
        int8  : 1250.40
        int16 : 1225.47
  
      Transfer bandwidth (GBPS)
        enqueueWriteBuffer              : 15.31
        enqueueReadBuffer               : 14.64
        enqueueWriteBuffer non-blocking : 31.70
        enqueueReadBuffer non-blocking  : 28.31
        enqueueMapBuffer(for read)      : 22.14
          memcpy from mapped ptr        : 14.49
        enqueueUnmap(after write)       : 34.69
          memcpy to mapped ptr          : 15.92
  
      Kernel launch latency : 35.80 us
  ```

- 项目中使用了3种方案来实现矩阵乘法，并调用了`Intel MKL`的接口来进行性能对比，最后利用子矩阵和改变矩阵存储顺序的优化取得了巨大的性能提升，测试结果如下：

  |           | 100*100 |    1K*1K |    5K*5K |   10K*10K |
  | :--------: | ------: | -------: | -------: | --------: |
  |   My Code | 0.113 ms |  2.06 ms | 229.23 ms | 1732.81 ms |
  | Intel MKL | 9.264 ms | 23.64 ms | 775.32 ms | 3773.68 ms |

- 另外还实现了矩阵的卷积运算，测试结果如下

  <img src=".\img\matrixconv.png" alt="matrixconv" width=400 height=200 />
