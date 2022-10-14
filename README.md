# YOLOv5-Light

这个仓库记录了一些对YOLOv5的魔改，如结构的替换、通道修剪和量化等模型压缩操作。

## Requirements

```bash
pip3 install -r requirements.txt
```

## Architecture

### ConvNeXt

ConvNeXt作为22年新提出的Conv网络，吸收了ViT的一些思想构建而成，相比先进的Swin等也有很好的精度。

这里将ConvNeXt的结构分为Stem，Block和Downsample几个组件分别实现。各组件的参数设置也与YOLOv5原版代码尽量保持一致。详见`models/yolov5_convnexts.yaml`

由于重新解构了组件，ConvNeXt的预训练权重无法直接加载，需要修改一下各项的名字。

```bash
python train.py --data data/coco.yaml --cfg models/yolov5_convnexts.yaml --weights pretrained_weights/convnext/convnext_small_22k_224.pth --batch-size 16 --hyp data/hyps/hyp.scratch-convnext.yaml --optimizer AdamW
```

（这里的超参有很大的修改空间，实验中也发现默认参数训练起来很容易发散）

## Channel Pruning

YOLOv5本身的结构相当紧凑，而且提供了多个放缩模型，通道修剪的意义不算大。结合v5的设计思想，设计了两种压缩方法：

### Search-based

为避免修改CSP Net的思想和保持结构的连续，只对Conv模块，C3模块的cv3和bottleneck和SPPF模块的cv2部分执行修剪。对上述要压缩的部分设置一个缩放系数，可以在`models/yolov5l_slim.yaml`中看到。

核心的修剪算法来自EagleEye(ECCV20)，通过随机大量搜索+BN重校准的方式搜索子网结构。

执行下面的命令完成修剪：

```bash
python pruning/ee.py --weights pretrained_weights/yolov5l.pt --cfg models/yolov5l_slim.yaml --data data/coco.yaml --device 0 --saved_cfg models/yolov5l_slimmed.yaml --remain_ratio 0.5
```

会保存压缩后的模型配置文件和权重参数，需要**重新训练**才能得到比较好的效果。

### Regularization-based

YOLOv3的压缩算法大多基于Network Slimming，通过对BN层缩放因子施加L1正则化，迫使模型鉴别出一些不重要的通道。但对于v5来说，稀疏训练比较困难，要想达到比较好的压缩效果，需要反复调整稀疏训练的参数。

执行下面的命令完成稀疏训练和修剪：

```bash
python pruning/train_sparsity.py --data data/coco.yaml --weights pretrained_weights/yolov5l.pt --cfg models/yolov5l_slim.yaml --batch-size 64 --st --sr 0.002
```

默认稀疏率sr线性减小，可以通过tensorboard观察BN缩放因子的分布情况，选择合适的阈值修剪。

```bash
python pruning/slim.py --weights runs/train/yolov5n_hand_s0.002/weights/best.pt --cfg models/yolov5n_slim.yaml --data data/hand.yaml --device 0 --saved_cfg models/yolov5n_slimmed.yaml --thresh 0.00001
```

修剪结束会评估精度，同样建议微调。

## Quantization

这部分采用成熟的量化工具[PPQ](https://github.com/openppl-public/ppq)来实现功能。

对于训练好的pt文件，先导出为ONNX：

```bash
python export.py --weights pretrained_weights/yolov5l.pt --data data/coco.yaml --include onnx
```

然后执行下面的命令完成量化：

```bash
python quantization/ppq_quantize.py
```

脚本里提供了LSQ、AdaRound和TRT的量化实现。自行修改默认参数即可，很方便。对于前两者需要对weights做微调，因此校准集太大容易CUDA OOM。
实验发现LSQ和TRT效果比较好，AdaRound最后的信噪比有点高。