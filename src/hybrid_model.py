import torch
import torch.nn as nn
import timm

class CNN_Swin_Transformer_FC(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_Swin_Transformer_FC, self).__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(768 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.upsample(x)
        x = self.swin_transformer.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.flatten(x)
        return self.fc_block(x)
