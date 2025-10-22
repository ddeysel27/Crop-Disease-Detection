import timm, torch, torch.nn as nn

def build(num_classes: int, name: str = "vit_base_patch16_224", pretrained: bool=True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
