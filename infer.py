import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset,get_lettuce_dataset
from losses import get_losses
from extend_sam import get_model, get_optimizer, get_scheduler, get_opt_pamams, get_runner
import random
import numpy as np
import torch
import os
import glob
from torchvision import transforms
from PIL import Image
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
    test_cfg = config.test
    train_cfg = config.train
    model = get_model(model_name=train_cfg.model.sam_name, **train_cfg.model.params)
    #我以./weed_data下的图片为例
    test_data_path = '../weed_data'
    output_dir = './result'
    model.load_state_dict(torch.load('/root/autodl-tmp/sam_finetune/finetune-anything/experiment/model/semantic_sam/model.pth'))
    model.eval()
    model.cuda()
    preprocess = transforms.Compose([
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                    ])
    for img_path in glob.glob(os.path.join(test_data_path,'*.jpg')):
        print(img_path)
        # 读取图片并转换为 RGB
        img = Image.open(img_path).convert("RGB")
        # 对图片进行预处理，并增加 batch 维度
        input_tensor = preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
            output,_ = model(input_tensor)
        pred = torch.argmax(output, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        sub_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        unique_labels = np.unique(pred)
        for label in unique_labels:
            # 生成二值 mask：像素属于该类别则为 255，否则为 0
            binary_mask = (pred == label).astype(np.uint8) * 255
            mask_filename = os.path.join(sub_output_dir, f'class_{label}.png')
            cv2.imwrite(mask_filename, binary_mask)
        print(f"Processed {img_path}, saved masks to {sub_output_dir}")