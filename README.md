# oneAPI学习笔记
- 此项目记录的是[`Intel oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)的学习和测试代码，我的机器使用的CPU是`Intel Ultra 5 125H`，它是一款集成了GPU的SOC。

  <img src=".\img\machine.png" alt="machine" width=400 height=200 />

- 项目中使用了3种方案来实现矩阵乘法，并调用了`Intel MKL`的接口来进行性能对比，最后利用子矩阵和改变矩阵存储顺序的优化取得了巨大的性能提升，测试结果如下：

  |           | 100*100 |    1K*1K |    5K*5K |   10K*10K |
  | :--------: | ------: | -------: | -------: | --------: |
  |   My Code | 0.129 ms |  2.67 ms | 257.81 ms | 2225.03 ms |
  | Intel MKL | 7.649 ms | 13.48 ms | 502.82 ms | 3083.07 ms |

  | <img src=".\img\100MatrixMulti.png" alt="100MatrixMulti" width=400 height=200 align=center /> | <img src=".\img\1KMatrixMulti.png" alt="1KMatrixMulti" width=400 height=200 align=center /> |
  |------------|------------|
  | <img src=".\img\5KMatrixMulti.png" alt="5KMatrixMulti" width=400 height=200 align=center /> | <img src=".\img\10KMatrixMulti.png" alt="10KMatrixMulti" width=400 height=200 align=center /> |


- 另外还实现了矩阵的卷积运算，测试结果如下

  <img src=".\img\matrixconv.png" alt="matrixconv" width=400 height=200 />
