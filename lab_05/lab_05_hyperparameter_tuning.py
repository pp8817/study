#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import wandb

from training_utilities import train_loop, evaluation_loop, save_checkpoint, load_checkpoint


# 이번 실습시간에는 다양한 학습 전략과 hyperparameter tuning을 통해 CIFAR-10 테스트셋에서 높은 분류 성능을 얻는 것이 목표이다.
# 
# <mark>과제</mark> 다양한 조건에서 CIFAR-10 데이터셋 학습을 실험해보고 test 데이터셋에서 80% 이상의 accuracy를 달성하라.
# 
# * 제출물1 : <u>5개 이상의 학습 커브</u>를 포함하는 wandb 화면 캡처 (wandb 웹페이지의 본인 이름 포함하여 캡처)
# * 제출물2 : 실험 결과에 대한 분석과 논의 (아래에 markdown으로 기입)
# 
# 참고: 코드에 대한 pytest가 따로 없으므로 자유롭게 코드를 변경하여도 무방함.
# 
# 단, <U>Transfer learning 혹은 Batch size는 변경은 수행하지 말것</U>
# 
# 실험 조건 예시
# - [Network architectures](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
# - input normalization
# - [Weight initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_)
# - [Optimizers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) (Adam, SGD with momentum, ... )
# - Regularizations (weight decay, dropout, [Data augmentation](https://pytorch.org/vision/0.9/transforms.html), ensembles, ...)
# - learning rate & [learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
# 
# 스스로 neural network를 구축할 경우 아래 사항들을 고려하라
# - Filter size
# - Number of filters
# - Pooling vs Strided Convolution
# - Activation functions

# In[2]:


def get_model(model_name, num_classes, config):
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, num_classes)

        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    else:
        raise Exception("Model not supported: {}".format(model_name))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Using model {model_name} with {total_params} parameters ({trainable_params} trainable)")

    return model


# In[3]:


def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42

    normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
    
    # train_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # 데이터 전처리
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)

    # Split train dataset into train and validataion dataset
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), 
                                                  test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # DataLoader
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_worker, **kwargs)
    
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes


# In[4]:


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # 변경된 부분
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  # 변경된 부분

def train_main(config):
    ## data and preprocessing settings
    data_root_dir = config['data_root_dir']
    num_worker = config.get('num_worker', 4)

    ## Hyper parameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    start_epoch = config.get('start_epoch', 0)
    num_epochs = config['num_epochs']

    ## checkpoint setting
    checkpoint_save_interval = config.get('checkpoint_save_interval', 10)
    checkpoint_path = config.get('checkpoint_path', "checkpoints/checkpoint.pth")
    best_model_path = config.get('best_model_path', "checkpoints/best_model.pth")
    load_from_checkpoint = config.get('load_from_checkpoint', None)

    ## variables
    best_acc1 = 0

    wandb.init(
        project=config["wandb_project_name"],
        config=config
    )

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataloader, val_dataloader, test_dataloader, num_classes = load_cifar10_dataloaders(
        data_root_dir, device, batch_size = batch_size, num_worker = num_worker)
    
    model = get_model(model_name = config["model_name"], num_classes= num_classes, config = config).to(device)
    initialize_weights(model)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # optimizer, 변경 가능 부분, 1
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) # 2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) # 3
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # lr, 변경 가능 부분

    if load_from_checkpoint:
        load_checkpoint_path = (best_model_path if load_from_checkpoint == "best" else checkpoint_path)
        start_epoch, best_acc1 = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, device)

    if config.get('test_mode', False):
        # Only evaluate on the test dataset
        print("Running test evaluation...")
        test_acc = evaluation_loop(model, device, test_dataloader, criterion, phase = "test")
        print(f"Test Accuracy: {test_acc}")
        
    else:
        # Train and validate using train/val datasets
        for epoch in range(start_epoch, num_epochs):
            train_loop(model, device, train_dataloader, criterion, optimizer, epoch)
            val_acc1 = evaluation_loop(model, device, val_dataloader, criterion, epoch = epoch, phase = "validation")
            scheduler.step()

            if (epoch + 1) % checkpoint_save_interval == 0 or (epoch + 1) == num_epochs:
                is_best = val_acc1 > best_acc1
                best_acc1 = max(val_acc1, best_acc1)
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_acc1, is_best, best_model_path)

    wandb.finish()


# 실험이 모두 끝나면 best model에 대해 test set성능을 평가한다. 

# In[6]:


config_testmode = {
    **config, 
    'test_mode': True, # True if evaluating only test set
    'load_from_checkpoint': 'best'
}

train_main(config_testmode)


# <mark>제출물</mark>
# 
# 1. 본인 이름이 나오도록 wandb 결과 화면을 캡처하여 `YOUR_PRIVATE_REPOSITORY_NAME/lab_05/wandb_results.png`에 저장한다. (5 points)
# 2. 결과를 table로 정리한 뒤 그 아래에 분석 및 논의를 작성 한다. (15 points)
# 
# -----

# #### wandb 결과
# 
# <center><img src="./wandb_results.png" width="1000px"></img></center>
# 
# #### 5개 이상의 실험 결과
# 
# | 모델 | 실험 조건 | val_accuracy | 설명  |
# |------|----------|--------------|------|
# | resnet50   | optimizer: SGD, lr: 1e-3, epochs: 100     |      50.5      |   Default SGD만 적용한 기본 모델   |
# |  resnet50   | optimizer:SGD(momentum=0.9 , weight decay), lr: 1e-3, epochs: 100          |        58.79      |  SGD에 모멘텀을 추가    |
# |   vgg16   |    optimizer:SGD(momentum=0.9, weight decay), lr: 1e-3, data proccessing, epochs: 100      |   86.4           |    resnet에서 vgg16으로 모델 변경, 데이터 전처리 적용  |
# |  vgg16    |    optimizer:SGD(momentum=0.9, weight decay), lr: 1e-3, data proccessing, Leaky ReLU, epochs: 100     | 86.5             |   Leaky ReLU 적용  |
# |  vgg16    |    optimizer:SGD(momentum=0.9, weight decay), lr: 1e-3, data proccessing, actication, Weight initialization, epochs: 100       |       86.89      |   가중치 초기화 적용   |
# 
# best model test_set accuracy: 
# 
# #### 분석 및 논의
# optimizer의 변화가 아닌 Data Proccessing, 활성화 함수 변경 등을 통해 validation accuracy를 높이고 싶어서 모든 optimizer는 SGD로 통일했습니다.
# 처음에는 아무것도 적용하지 않은 SGD로 테스트를 했고 50%라는 낮은 정확도를 기록했습니다.
# 정확도를 높이기 위해 SGD에 momentum을 0.9로 적용해 테스트를 했고 약 59% 정도를 기록했지만, 기대한 바와 다르게 정확도는 8% 정도만 상승했습니다.
# 방향성을 다르게 해서 모델을 resnet50에서 vgg16으로 변경하고, 데이터 전처리를 진행해 데이터의 다양성을 높여서 테스트를 진행한 결과 정확도가 86.4%로 크게 상승한 것을 확인했습니다. 데이터 전처리 후 정확도가 크게 증가하는 것으로 보아 데이터의 다양성이 필요했던 것이 아닐까 생각했습니다.
# 이후 정확도를 더욱 높이기 위해 활성화 함수를 Leaky ReLU로 변경, 가중치 초기화 등을 적용했지만 정확도는 3번째 테스트와 유사하게 나왔습니다.
# 
# 결론적으로 정확도 상승에 큰 기여를 한 것은 데이터 전처리가 아닐까 생각됩니다.

# -----
# #### Lab을 마무리 짓기 전 저장된 checkpoint를 모두 지워 저장공간을 확보한다

# In[7]:


import shutil, os
if os.path.exists('checkpoints/'):
    shutil.rmtree('checkpoints/')


# In[ ]:




