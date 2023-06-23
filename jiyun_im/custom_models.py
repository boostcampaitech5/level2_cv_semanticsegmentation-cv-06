import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from typing import Dict

NUM_CLASSES = 29

#print(torch.__version__)
#model = models.segmentation.fcn_resnet50(weights=None, progress=True, num_classes=29, weights_backbone="ResNet50_Weights.IMAGENET1K_V1")  ## aux_loss로 멀티모달..?
#print(model)

# _utils.py
class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

# 비교용
def fcn_res50():
    backbone = models.resnet50()

def fcn_swinv2t():
    backbone = models.swin_v2_t(weights="Swin_V2_T_Weights.IMAGENET1K_V1")

    return_layers = {"permute": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(768, 192, 3, padding=1, bias=False),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(192, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model

# efficientnet
def fcn_effb7():
    backbone = models.efficientnet_b7(weights="EfficientNet_B7_Weights.IMAGENET1K_V1")
    return_layers = {"features": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(2560, 1280, 3, padding=1, bias=False),
        nn.BatchNorm2d(1280),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(1280, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model

def fcn_effv2s():
    backbone = models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")

    return_layers = {"features": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(1280, 640, 3, padding=1, bias=False),
        nn.BatchNorm2d(640),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(640, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model

def fcn_effv2l():
    backbone = models.efficientnet_v2_l(weights="EfficientNet_V2_L_Weights.IMAGENET1K_V1")

    return_layers = {"features": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(1280, 640, 3, padding=1, bias=False),
        nn.BatchNorm2d(640),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(640, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model

# densenet
def fcn_den121():
    backbone = models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")

    return_layers = {"features": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(1024, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model

#mobilenet
def fcn_mobv2():
    backbone = models.mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V2")

    return_layers = {"features": "out"} #fc 전 layer
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = nn.Sequential(
        nn.Conv2d(1280, 640, 3, padding=1, bias=False),
        nn.BatchNorm2d(640),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(640, NUM_CLASSES, 1),
    )
    model = models.segmentation.FCN(backbone, classifier)

    return model



if __name__=="__main__":
    #eff = fcn_effb7()
    #print(eff)
    #den = fcn_den121()
    #print(den)
    mob = fcn_mobv2()
    print(mob)