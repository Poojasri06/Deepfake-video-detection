# src/model.py
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 frame encoder + BiLSTM temporal model + classification head.
    Input shape for forward: (B, T, C, H, W)
    Output: logits (B, 2)
    """
    def __init__(self, cnn_out_dim=256, lstm_hidden=128, bidirectional=True):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # adapt classifier to produce features
        if isinstance(backbone.classifier, nn.Sequential):
            in_feats = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_feats, cnn_out_dim)
        else:
            in_feats = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, cnn_out_dim))

        self.cnn = backbone
        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_hidden * (2 if bidirectional else 1), 2)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)                # (B*T, cnn_out_dim)
        feats = feats.view(B, T, -1)       # (B, T, feat)
        out, _ = self.lstm(feats)          # (B, T, hid*dirs)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits
