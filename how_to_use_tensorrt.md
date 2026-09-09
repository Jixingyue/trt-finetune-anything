# 模型导出与推理

## 1. ONNX模型导出与推理

### 1.1 导出ONNX模型
我们首先将PyTorch模型导出为ONNX格式。主要步骤包括：

1. 加载训练好的PyTorch模型
2. 准备示例输入（dummy input）
3. 使用torch.onnx.export导出ONNX模型，设置：
   - 输入输出节点名称
   - ONNX算子集版本
   - 动态batch size支持
   - 常量折叠优化等

### 1.2 ONNX推理
使用ONNX Runtime进行模型推理，流程如下：

1. 加载ONNX模型
2. 图像预处理：
   - 读取图像并转RGB
   - resize到1024x1024
   - 转换为所需的输入格式
3. 执行推理
4. 后处理：
   - 获取预测类别
   - 生成每个类别的二值掩码
   - 保存结果

### 1.3 性能评估
- 记录每张图片的推理时间
- 输出处理结果和保存路径

## 2. TensorRT模型转换与推理

### 2.1 ONNX转TensorRT
使用TensorRT自带的工具进行转换：
1. 使用TensorRT安装目录下的`/usr/src/tensorrt/bin`中的工具
2. 直接将ONNX模型转换为TensorRT的engine文件
3. 可以指定精度（FP32/FP16/INT8）和其他优化参数
4. 执行命令：
```bash
trtexec --onnx=best.onnx --saveEngine=best.engine  --fp16
```

### 2.2 TensorRT推理
TensorRT推理流程如下：

1. 加载TensorRT引擎：
   - 使用`trt.Runtime`创建运行时环境
   - 从文件读取序列化的引擎数据
   - 使用`deserialize_cuda_engine`反序列化引擎
   - 创建执行上下文`context`

2. 内存管理：
   - 为输入输出分配主机内存和设备内存
   - 使用`HostDeviceMem`类管理内存映射
   - 通过`cuda.pagelocked_empty`和`cuda.mem_alloc`分配内存

3. 图像预处理：
   - 读取图像并转RGB
   - resize到1024x1024
   - 转换为NCHW格式的numpy数组
   - 将数据复制到输入缓冲区

4. 执行推理：
   - 将输入数据从CPU传输到GPU
   - 设置输入输出张量地址
   - 调用`execute_async_v3`执行异步推理
   - 将结果从GPU传回CPU

5. 后处理：
   - 将输出数据重塑为正确的形状
   - 使用argmax获取预测类别
   - 为每个类别生成二值掩码
   - 保存结果到指定目录

### 2.3 性能对比
通过记录不同框架的推理时间，我们可以对比性能差异：

- PyTorch推理时间：通常作为基准，但速度较慢
- ONNX推理时间：比PyTorch有所提升，通常提高1.5-2倍
- TensorRT推理时间：最快，从测试结果看，每张图像处理时间可以达到毫秒级别，比PyTorch快3-5倍

TensorRT的优势在于：
- 针对NVIDIA GPU的深度优化
- 支持FP16/INT8量化，进一步提升性能
- 内核融合和内存优化，减少数据传输开销
- 动态内存管理，提高GPU利用率
