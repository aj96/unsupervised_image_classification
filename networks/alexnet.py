import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.alexnet import AlexNet
import torch.utils.model_zoo as model_zoo

def alexnet(num_classes, pretrained=False):
    """
    Link to pytorch alexnet implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

    Args:
        num_layers (int): number of layers for resnet
        num_classes (int): number of output classes
        pretrained (bool): Whether or not to use pretrained imagenet weights

    Returns:
        The AlexNet model (torchvision.models.alexnet.AlexNet)    
    """

    model = AlexNet(num_classes=num_classes)

    if pretrained:
        model_urls = {
            'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        }
        pretrained_dict = models.utils.load_state_dict_from_url(model_urls['alexnet'],
                                              progress=True)

        # Only load pretrained image-net weights for layers which are not final fc layer
        # as final fc layer has shape mis-match
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier.6' not in k}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key in pretrained_dict.keys():
                model_dict[key] = pretrained_dict[key]
        model.load_state_dict(model_dict)
    return model
        
if __name__ == "__main__":
    model = alexnet(num_classes=10, pretrained=True)
    print(model)
