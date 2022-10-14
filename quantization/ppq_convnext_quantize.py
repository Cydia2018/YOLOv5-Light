from typing import Iterable
import os

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from ppq import *
from ppq.api import *
from ppq.executor.torch import TorchExecutor

# ------------------------------------------------------------
# 在 PPQ 中我们目前提供两种不同的算法帮助你微调网络
# 这些算法将使用 calibration dataset 中的数据，对网络权重展开重训练
# 1. 经过训练的网络不保证中间结果与原来能够对齐，在进行误差分析时你需要注意这一点
# 2. 在训练中使用 with ENABLE_CUDA_KERNEL(): 子句将显著加速训练过程
# 3. 训练过程的缓存数据将被贮存在 gpu 上，这可能导致你显存溢出，你可以修改参数将缓存设备改为 cpu
# ------------------------------------------------------------
'''
# ------------------------------------------------------------
# yolov5 trt
# ------------------------------------------------------------
ONNX_PATH        = '../runs/train/exp2/weights/yolov5_convnexts.onnx'       # 你的模型位置
ENGINE_PATH      = '../runs/train/exp2/weights/yolov5_convnexts_trt.onnx'  # 生成的 Engine 位置
CALIBRATION_PATH = '/home/lxc/bishe/calib_hand_images'                         # 校准数据集
BATCHSIZE        = 1
EXECUTING_DEVICE = 'cuda'
# create dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([640, 640]),  # [h,w]
    transforms.ToTensor(),
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img) # img is 0 - 1

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)

qir = quantize_onnx_model(
    platform=TargetPlatform.TRT_INT8,
    onnx_import_file=ONNX_PATH, 
    calib_dataloader=dataloader, 
    calib_steps=32, device=EXECUTING_DEVICE,
    input_shape=[BATCHSIZE, 3, 640, 640], 
    collate_fn=lambda x: x.to(EXECUTING_DEVICE))

snr_report = graphwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

snr_report = layerwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

export_ppq_graph(
    qir, platform=TargetPlatform.TRT_INT8, 
    graph_save_to=ENGINE_PATH)
'''


# ------------------------------------------------------------
# yolov5 lsq
# ------------------------------------------------------------
ONNX_PATH        = '../runs/train/exp2/weights/yolov5_convnexts.onnx'       # 你的模型位置
ENGINE_PATH      = '../runs/train/exp2/weights/yolov5_convnexts_trt_lsq.onnx'  # 生成的 Engine 位置
CALIBRATION_PATH = '/home/lxc/bishe/calib_hand_images'                         # 校准数据集
BATCHSIZE        = 1
w                = 640
h                = 640
INPUT_SHAPE      = [BATCHSIZE, 3, h, w]
EXECUTING_DEVICE = 'cuda'
PLATFORM         = TargetPlatform.TRT_INT8
# create dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([h, w]),  # [h,w]
    transforms.ToTensor(),
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img) # img is 0 - 1

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)

QSetting = QuantizationSettingFactory.default_setting()
QSetting.lsq_optimization                            = True
QSetting.lsq_optimization_setting.block_size         = 4
QSetting.lsq_optimization_setting.lr                 = 1e-5
QSetting.lsq_optimization_setting.gamma              = 0
QSetting.lsq_optimization_setting.is_scale_trainable = True
QSetting.lsq_optimization_setting.collecting_device  = 'cuda'

qir = quantize_onnx_model(
    platform=PLATFORM,
    onnx_import_file=ONNX_PATH, 
    calib_dataloader=dataloader, 
    calib_steps=32, device=EXECUTING_DEVICE,
    input_shape=INPUT_SHAPE, 
    collate_fn=lambda x: x.to(EXECUTING_DEVICE),
    setting=QSetting)

snr_report = graphwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

snr_report = layerwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

# export_ppq_graph(
#     qir, platform=TargetPlatform.TRT_INT8, 
#     graph_save_to=ENGINE_PATH)


'''
# ------------------------------------------------------------
# yolov5 adaround
# ------------------------------------------------------------
ONNX_PATH        = 'pretrained_weights/yolov5s.onnx'       # 你的模型位置
ENGINE_PATH      = 'pretrained_weights/yolov5s_trt_cle_ada.onnx'  # 生成的 Engine 位置
CALIBRATION_PATH = '/home/lxc/bishe/calib_coco_medium_images'                         # 校准数据集
BATCHSIZE        = 1
w                = 640
h                = 640
INPUT_SHAPE      = [BATCHSIZE, 3, h, w]
EXECUTING_DEVICE = 'cuda'
PLATFORM         = TargetPlatform.TRT_INT8
# create dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([h, w]),  # [h,w]
    transforms.ToTensor(),
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img) # img is 0 - 1

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)

QSetting = QuantizationSettingFactory.default_setting()
QSetting.equalization = True
QSetting.quantize_parameter_setting.baking_parameter = False

qir = quantize_onnx_model(
    platform=PLATFORM,
    onnx_import_file=ONNX_PATH, 
    calib_dataloader=dataloader, 
    calib_steps=32, device=EXECUTING_DEVICE,
    input_shape=INPUT_SHAPE, 
    collate_fn=lambda x: x.to(EXECUTING_DEVICE),
    setting=QSetting)

from ppq.quantization.optim import AdaroundPass, ParameterBakingPass
executor = TorchExecutor(graph=qir, device=EXECUTING_DEVICE)
AdaroundPass(steps=5000).optimize(
    graph=qir, dataloader=dataloader, 
    executor=executor, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
ParameterBakingPass().optimize(
    graph=qir, dataloader=dataloader, 
    executor=executor, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

snr_report = graphwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

snr_report = layerwise_error_analyse(
    graph=qir, running_device=EXECUTING_DEVICE, 
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

export_ppq_graph(
    qir, platform=TargetPlatform.TRT_INT8, 
    graph_save_to=ENGINE_PATH)
'''
