# oneAPI学习笔记
- 此项目记录的是`Intel oneAPI`的学习和测试代码，我的机器使用的CPU是`Intel Ultra 5 125H`，它是一款集成了GPU的SOC。

  <img src=".\img\machine.png" alt="machine" style="zoom:50%;" />

- 项目中使用了3种方案来实现矩阵乘法，并调用了`Intel MKL`的接口来进行性能对比，最后得到了**24**倍左右的性能优势。

  <img src=".\img\matrixkernel_testresult.png" alt="matrixkernel_testresult" style="zoom:50%;" />

- 另外还实现了矩阵的卷积运算，测试结果如下

  <img src=".\img\matrixconv.png" alt="matrixconv" style="zoom:50%;" />
