import argparse
import time
from omegaconf import OmegaConf
import torch
from extend_sam import get_model
import onnxruntime
import numpy as np
import os
import glob
import cv2

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)
parser.add_argument('--cfg', default=None, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load("./config/{task_name}.yaml".format(task_name=args.task_name))
    
    train_cfg = config.train
    model = get_model(model_name=train_cfg.model.sam_name, **train_cfg.model.params)
    
    # 加载模型权重
    model.load_state_dict(torch.load('experiment/model/semantic_sam/model.pth', map_location='cpu'))
    model.eval()
    
    # 准备示例输入
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    # 确保模型和输入在同一设备上
    device = torch.device('cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,                     # 要转换的模型
        dummy_input,              # 模型的输入
        # "semantic_sam.onnx",      # 导出的ONNX文件名
        "best.onnx",      # 导出的ONNX文件名
        export_params=True,       # 存储训练好的参数权重
        opset_version=13,         # ONNX算子集版本
        do_constant_folding=True, # 是否执行常量折叠优化
        input_names=['input'],    # 输入节点的名称
        output_names=['output'],  # 输出节点的名称
        dynamic_axes={            # 动态尺寸
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("ONNX model exported successfully to semantic_sam.onnx")

    # 加载ONNX模型进行推理
    ort_session = onnxruntime.InferenceSession("semantic_sam.onnx")

    # 设置输入输出路径
    test_data_path = './weed_data'
    output_dir = './result2'

    # 遍历测试图片进行推理
    for img_path in glob.glob(os.path.join(test_data_path,'*.JPG')):
        print(img_path)
        # 读取图片并转换为RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理图片
        img = cv2.resize(img, (1024, 1024))
        input_tensor = img.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # ONNX推理
        start_time = time.time()  # Start time measurement
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0]
        end_time = time.time()  # End time measurement

        # 计算推理时间
        inference_time = end_time - start_time
        print(f"Inference time for {img_path}: {inference_time:.4f} seconds")

        # 后处理
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
