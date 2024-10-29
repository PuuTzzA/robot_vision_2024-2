 #%% Class Dataset
import torch
from torchvision import transforms
from PIL import Image
import json
import os

class ClassDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None, demo=False):
        self.folder = folder
        self.transform = transform if transform else transforms.PILToTensor()
        self.demo = demo
        self.annotations = self.load_annotations()
        self.images = self.load_images()

    def load_annotations(self):
        with open(os.path.join(self.folder, 'annotations.json')) as f:
            data = json.load(f)
        return data['annotations']
    
    def load_images(self):
        with open(os.path.join(self.folder, 'annotations.json')) as f:
            data = json.load(f)
        return data['images']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load the image
        img_info = self.images[idx]
        img_path = os.path.join(self.folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Convert to tensor
        if self.transform:
            image = F.to_tensor(image)
            # image = self.transform(image)
            
        # Load keypoints and bounding boxes
        gt_info = self.annotations[idx]
        keypoints = gt_info['keypoints']
        boxes = gt_info['boxes']
        labels = gt_info['labels']
        area = gt_info['area']
        iscrowd = gt_info['iscrowd']
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        
        return image, {
        "keypoints": keypoints,
        "boxes": boxes,
        "labels": labels,
        "area": area,
        "iscrowd": iscrowd
    }
        # return image, keypoints, boxes

#%%
import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import sys
sys.path.append("D:/RobotVision_3DCars/detection")

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate

#%%
def get_model(num_keypoints, weights_path=None):
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=((0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0),) * 5  # Aspect ratios for each size
    )
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=True,
        num_keypoints=num_keypoints,
        num_classes=2  # Background is the first class, object is the second class
    )

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model

# Setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Paths to dataset folders
KEYPOINTS_FOLDER_TRAIN = r"D:\RobotVision_3DCars\data_split\train\images_jpg"
KEYPOINTS_FOLDER_TEST = r"D:\RobotVision_3DCars\data_split\test\images_jpg"

# Create datasets
dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

# Create data loaders
data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Get the model
model = get_model(num_keypoints=14)
model.to(device)

# Setup optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2, verbose=True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device)

# Save model weights after training
torch.save(model.state_dict(), r"D:\RobotVision_3DCars\data_split\keypointrcnn_weights.pth")