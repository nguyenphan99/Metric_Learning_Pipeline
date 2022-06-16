import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
class Resnet50(nn.Module):
    def __init__(self, config):
        super(Resnet50, self).__init__()

        self.backbone = models.resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.path_model_pretrained))
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, config.num_embeddings)
        
    def forward(self, x):
        x = F.normalize(self.backbone(x))
        return x
class Resnet34(nn.Module):
    def __init__(self, config):
        super(Resnet34, self).__init__()

        self.backbone = models.resnet34(pretrained=False)
        # self.backbone.load_state_dict(torch.load(config.path_model_pretrained))
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, config.num_embeddings)
        
    def forward(self, x):
        x = F.normalize(self.backbone(x))
        return x
class Resnet18(nn.Module):
    def __init__(self, config):
        super(Resnet18, self).__init__()

        self.backbone = models.resnet18(pretrained=False)
        # self.backbone.load_state_dict(torch.load(config.path_model_pretrained))
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, config.num_embeddings)
        
    def forward(self, x):
        x = F.normalize(self.backbone(x))
        return x
class Resnet101(nn.Module):
    def __init__(self, config):
        super(Resnet101, self).__init__()

        self.backbone = models.resnet101(pretrained=False)
        # self.backbone.load_state_dict(torch.load(config.path_model_pretrained))
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, config.num_embeddings)
        
    def forward(self, x):
        x = F.normalize(self.backbone(x))
        return x
    
class Densenet121(nn.Module):
    def __init__(self, config):
        super(Densenet121, self).__init__()

        self.backbone = models.densenet121(pretrained=False)
        self.backbone.load_state_dict(torch.load(config.path_model_pretrained))
        
        in_feature = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_feature, config.num_embeddings)
        
    def forward(self, x):
        x = F.normalize(self.backbone(x))
        return x