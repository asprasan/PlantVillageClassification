import torch.nn as nn
from torchvision import models
from models import register_model

@register_model("efficientnet_v2_s")
class EfficientNet_V2_S(nn.Module):
    def __init__(self,
                 num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            )
        # modify model for number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280,
                       out_features=num_classes,
                       bias=True)
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

@register_model("efficientnet_v2_m")
class EfficientNet_V2_M(nn.Module):
    def __init__(self,
                 num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.efficientnet_v2_m(
            models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
            )
        # modify model for number of classes
        self.model.classifier = nn.Sequential(
            [nn.Dropout(p=0.2, inplace=True),
             nn.Linear(in_features=1280,
                       out_features=num_classes,
                       bias=True)]
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

