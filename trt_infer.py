import argparse
import time
import numpy as np
import os
import glob
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument('--engine_path', default='semantic_sam.engine', type=str, help='Path to TensorRT engine file')
parser.add_argument('--test_data_path', default='./weed_data', type=str, help='Path to test images')
parser.add_argument('--output_dir', default='./result_trt', type=str, help='Path to save results')

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name):
        self.host = host_mem
        self.device = device_mem
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    stream = cuda.Stream()
    
    # 使用TensorRT 10.x的API
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        
        # 分配CPU和GPU内存
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # 区分输入和输出
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem, name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, name))
    
    return inputs, outputs, stream

def do_inference(context, inputs, outputs, stream):
    # 传输输入数据到GPU
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    
    # 在TensorRT 10.x中，我们需要设置输入和输出缓冲区
    for inp in inputs:
        context.set_tensor_address(inp.name, int(inp.device))
    
    for out in outputs:
        context.set_tensor_address(out.name, int(out.device))
    
    # 执行推理 - 使用execute_async_v3
    context.execute_async_v3(stream_handle=stream.handle)
    
    # 从GPU传回预测结果
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    
    # 同步流
    stream.synchronize()
    
    # 仅返回主机输出
    return [out.host for out in outputs]

if __name__ == '__main__':
    args = parser.parse_args()
    
    # 加载TensorRT引擎
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(args.engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # 分配缓冲区
    inputs, outputs, stream = allocate_buffers(engine)
    
    # 设置输入输出路径
    test_data_path = args.test_data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历测试图片进行推理
    for img_path in glob.glob(os.path.join(test_data_path, '*.JPG')):
        print(img_path)
        # 读取图片并转换为RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理图片
        img = cv2.resize(img, (1024, 1024))
        input_tensor = img.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # 将输入数据复制到输入缓冲区
        np.copyto(inputs[0].host, input_tensor.ravel())
        
        # TensorRT推理
        start_time = time.time()
        trt_outputs = do_inference(context, inputs, outputs, stream)
        end_time = time.time()
        
        # 计算推理时间
        inference_time = end_time - start_time
        print(f"TensorRT inference time for {img_path}: {inference_time:.4f} seconds")
        
        # 后处理
        # 直接使用计算过的形状
        output_shape = (1, 2, 256, 256)  # 131072 / 4 = 32768 = 128 * 256
        print(f"Using output shape: {output_shape}")
        
        output = np.array(trt_outputs[0]).reshape(output_shape)
        
        # 使用argmax获取预测类别
        pred = np.argmax(output, axis=1)
        pred = pred.squeeze(0)
        
        # 保存结果
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        sub_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        
        unique_labels = np.unique(pred)
        for label in unique_labels:
            binary_mask = (pred == label).astype(np.uint8) * 255
            mask_filename = os.path.join(sub_output_dir, f'class_{label}.png')
            cv2.imwrite(mask_filename, binary_mask)
        
        print(f"Processed {img_path}, saved masks to {sub_output_dir}")
