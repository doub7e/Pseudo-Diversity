import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm

class ViTExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device)
        self.model.eval()
        
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Inception v3 requires input size of (299, 299)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization for pre-trained models
        ])
    
    def forward(self, x, detach=True):
        x = self.transform(x / 2 + 0.5)  # Generator output is in [-1, 1]
        feat = self.model(x)
        return feat.detach() if detach else feat



