import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import yaml
import contextlib
import thop
from terminaltables import AsciiTable

from models.yolo import Detect
from models.common import *
from models.experimental import *
from utils.general import make_divisible


def parse_module_defs(d):
    CBL_idx = []
    ignore_idx  =[]

    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    fromlayer = []
    from_to_map = {}
    ratio_list = []
    ratio_num_sum = 0

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m is SlimConv:
            ratio_list.append(args[1])
            ratio_num_sum += 1
            named_m_bn = named_m_base+'.bn'
            CBL_idx.append(named_m_bn)
            if i > 0:
                from_to_map[named_m_bn] = fromlayer[f]
            fromlayer.append(named_m_bn)
        elif m is SlimC3:
            ratio_list.append(args[1])
            ratio_num_sum += 1
            ratio_list.extend(args[2])
            ratio_num_sum += len(args[2])
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            named_m_cv3_bn = named_m_base + ".cv3.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = fromlayer[f]
            fromlayer.append(named_m_cv3_bn)
            c3fromlayer = [named_m_cv1_bn]

            ignore_idx.append(named_m_cv1_bn)
            ignore_idx.append(named_m_cv2_bn)
            CBL_idx.append(named_m_cv3_bn)
            for j in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(j)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(j)
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[j]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
                CBL_idx.append(named_m_bottle_cv1_bn)
                ignore_idx.append(named_m_bottle_cv2_bn)  # not prune shortcut
            from_to_map[named_m_cv3_bn] = [c3fromlayer[-1], named_m_cv2_bn]
        elif m is SlimSPPF:
            ratio_list.append(args[1])
            ratio_num_sum += 1
            named_m_cv1_bn = named_m_base+'.cv1.bn'
            named_m_cv2_bn = named_m_base+'.cv2.bn'
            CBL_idx.append(named_m_cv1_bn)
            ignore_idx.append(named_m_cv2_bn)
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn]*4
            fromlayer.append(named_m_cv2_bn)
        elif m is Concat:
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m is Detect:
            for j in range(3):
                ignore_idx.append(named_m_base + ".m.{}".format(j))
                from_to_map[named_m_base + ".m.{}".format(j)] = fromlayer[f[j]]
        else:
            fromlayer.append(fromlayer[-1])

    return ignore_idx, from_to_map, ratio_num_sum


def obtain_filtermask_l1(conv_module, rand_remain_ratio):
    w_copy = conv_module.weight.data.abs().clone()
    w_copy = torch.sum(w_copy, dim=(1,2,3))
    length = w_copy.size(0)
    num_retain = max(make_divisible(int(length * rand_remain_ratio), 8), 8)
    _, indice = torch.topk(w_copy, num_retain)
    mask = torch.zeros(length)
    mask[indice.cpu()] = 1

    return mask

def obtain_filtermask_bn(bn_module, thresh):
    w_copy = bn_module.weight.data.abs().clone()
    num_retain = w_copy.gt(thresh).sum().item()
    length = w_copy.size(0)
    num_retain = max(make_divisible(num_retain, 8), 8)
    _, index = torch.topk(w_copy, num_retain)
    mask = torch.zeros(length)
    mask[index.cpu()] = 1

    return mask


def gather_bnweights(model, ignore_idx):
    size_list = []
    weights_list = []
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and (n not in ignore_idx):
            size_list.append(m.weight.data.size(0))
            weights_list.append(m.weight.data.abs().clone())
    
    bn_weights = torch.zeros(sum(size_list))
    idx = 0
    for w, size in zip(weights_list, size_list):
        bn_weights[idx:(idx + size)] = w
        idx += size
    return bn_weights


def weights_inheritance(model, compact_model, from_to_map, maskbndict):
    modelstate = model.state_dict()
    pruned_model_state = compact_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    
    last_idx = 0  # find last layer index
    for (layername, layer) in model.named_modules():
        try:
            last_idx = max(last_idx, int(layername.split('.')[1]))
        except:
            pass

    for ((layername, layer),(pruned_layername, pruned_layer)) in zip(model.named_modules(), compact_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith(f"model.{last_idx}"):
            convname = layername[:-4] + "bn"

            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) ==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w
                    pruned_layer.out_channels = w.size(0)
                    pruned_layer.in_channels = w.size(1)
                if isinstance(former, list):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))

                    former_kernel_num = [modelstate[former_item + ".weight"].shape[0] for former_item in former]
                    in_idx = []
                    for i in range(len(former)):
                        former_item = former[i]
                        in_idx_each = np.squeeze(np.argwhere(np.asarray(maskbndict[former_item].cpu().numpy())))

                        if i > 0:
                            in_idx_each = [k + sum(former_kernel_num[:i]) for k in in_idx_each]
                        in_idx.extend(in_idx_each)
   
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) ==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w
                    pruned_layer.out_channels = w.size(0)
                    pruned_layer.in_channels = w.size(1)

            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                if len(w.shape) == 3:
                    w = w.unsqueeze(0)
                pruned_layer.weight.data = w
                pruned_layer.out_channels = w.size(0)
                pruned_layer.in_channels = w.size(1)

        if isinstance(layer, nn.Conv2d) and layername.startswith(f"model.{last_idx}"):  # --------------------------------
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :].clone()
            pruned_layer.bias.data = layer.bias.data.clone()
            pruned_layer.out_channels = w.size(0)
            pruned_layer.in_channels = w.size(1)

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            pruned_layer.num_features = w.size(0)

    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    cm = compact_model.module.model[-1] if hasattr(compact_model, 'module') else compact_model.model[-1]
    cm.anchors = m.anchors.clone()
    compact_model.nc = model.nc


def update_yaml_loop(d, name, maskconvdict):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m is SlimConv:
            c1, c2 = ch[f], args[0]
            ratio = args[1]
            if c2 != no:  # if not output
                c2 = max(make_divisible(c2 * gw * ratio, 8), 8)
            named_m_conv = named_m_base+'.conv'
            if name == named_m_conv:
                args[1] = maskconvdict[named_m_conv].sum().item() / c2
        elif m is SlimSPPF:
            c1, c2 = ch[f], args[0]
            ratio = args[1]
            if c2 != no:  # if not output
                c2 = max(make_divisible(c2 * gw * ratio, 8), 8)
            named_m_cv1_conv = named_m_base+'.cv1.conv'
            if name == named_m_cv1_conv:
                args[1] = 0.5 * maskconvdict[named_m_cv1_conv].sum().item() / c2
        elif m is SlimC3:
            c1, c2 = ch[f], args[0]
            ratio = args[1]
            c2_ = make_divisible(c2 * gw, 8)
            if c2 != no:  # if not output
                c2 = max(make_divisible(c2 * gw * ratio, 8), 8)
            named_m_cv1_conv = named_m_base + ".cv1.conv"
            named_m_cv2_conv = named_m_base + ".cv2.conv"
            named_m_cv3_conv = named_m_base + ".cv3.conv"
            if name == named_m_cv1_conv:
                continue
            elif name == named_m_cv2_conv:
                continue
            elif name == named_m_cv3_conv:
                args[1] = maskconvdict[named_m_cv3_conv].sum().item() / c2
                continue
            for j in range(n):
                named_m_bottle_cv1_conv = named_m_base + ".m.{}.cv1.conv".format(j)
                if name == named_m_bottle_cv1_conv:
                    args[2][j] = maskconvdict[named_m_bottle_cv1_conv].sum().item() / (c2_*0.5)
                    continue


def update_yaml(pruned_yaml, model, ignore_conv_idx, maskdict, opt):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name not in ignore_conv_idx:
                update_yaml_loop(pruned_yaml,name,maskdict)
    
    return pruned_yaml


def obtain_quantiles(bn_weights, num_quantile=5):
    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)
    return quantiles


def model_flops(model, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters

    p = next(model.parameters())
    stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
    im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
    flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
    imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    fs = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs

    return fs