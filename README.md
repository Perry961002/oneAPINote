# oneAPI学习笔记
- 此项目记录的是`Intel oneAPI`的学习和测试代码，我的机器使用的CPU是`Intel Ultra 5 125H`，它是一款集成了GPU的SOC。

  <img src=".\img\machine.png" alt="machine" style="zoom:50%;" />

- 项目中使用了3种方案来实现矩阵乘法，并调用了`Intel MKL`的接口来进行性能对比，最后利用子矩阵和改变矩阵存储顺序的优化取得了20%~60%的性能提升。

  <img src=".\img\small_matrixmulti.png" alt="small_matrixmulti" style="zoom:50%;" />

  <img src=".\img\big_matrixmulti.png" alt="big_matrixmulti" style="zoom:50%;" />

- 另外还实现了矩阵的卷积运算，测试结果如下

  <img src=".\img\matrixconv.png" alt="matrixconv" style="zoom:50%;" />
