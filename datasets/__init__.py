from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset, TorchVOCSegmentation, LettuceSegDataset
from .transforms import get_transforms
from torchvision.datasets import VOCSegmentation
from sklearn.model_selection import train_test_split
import glob
import os
segment_datasets = {'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
                    'voc_sem': VOCSemanticDataset, 'torch_voc_sem': TorchVOCSegmentation, 'lettuce_sem':LettuceSegDataset}
det_dataset = {'base_det': BaseDetectionDataset, }
def get_lettuce_dataset():
    all_image_paths = sorted(glob.glob(os.path.join('/root/autodl-tmp/lettuce', "*.JPG")))
    train_image_paths, val_image_paths = train_test_split(
        all_image_paths, test_size=0.2, random_state=42
    )
    print(f"训练集数量: {len(train_image_paths)}")
    print(f"测试集数量: {len(val_image_paths)}")
    train_dataset = LettuceSegDataset(train_image_paths, width=1024, height=1024)
    val_dataset   = LettuceSegDataset(val_image_paths, width=1024, height=1024)
    return train_dataset,val_dataset
def get_dataset(cfg):
    name = cfg.name
    assert name in segment_datasets or name in det_dataset, \
        print('{name} is not supported, please implement it first.'.format(name=name))
    # TODO customized dataset params:
    # customized dataset params example:
    # if xxx:
    #   param1 = cfg.xxx
    #   param2 = cfg.xxx
    # return name_dict[name](path, model, param1, param2, ...)
    transform = get_transforms(cfg.transforms)
    if name in det_dataset:
        return det_dataset[name](**cfg.params, transform=transform)
    target_transform = get_transforms(cfg.target_transforms)
    return segment_datasets[name](**cfg.params, transform=transform, target_transform=target_transform)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data
