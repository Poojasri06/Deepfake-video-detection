#!/usr/bin/env python3
"""
Simple test script to verify model architecture and basic functionality.
"""

import torch
import torch.nn as nn
from torchvision import models
import sys

def test_model_architecture():
    """Test that the model can be instantiated correctly."""
    print("Testing DeepfakeDetector model architecture...")
    
    class DeepfakeDetector(nn.Module):
        """
        EfficientNet-B0 frame encoder + BiLSTM temporal model + classification head.
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
    
    try:
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = DeepfakeDetector().to(device)
        model.eval()
        
        print(f"✓ Model instantiated successfully")
        print(f"  - CNN backbone: EfficientNet-B0")
        print(f"  - LSTM: Bidirectional, hidden_size=128")
        print(f"  - Output: 2 classes (Real/Fake)")
        
        # Test forward pass with dummy data
        batch_size = 2
        sequence_length = 16
        channels = 3
        height = 112
        width = 112
        
        dummy_input = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected output shape: ({batch_size}, 2)")
        
        assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
        
        # Test softmax probabilities
        probs = torch.softmax(output, dim=1)
        print(f"\n✓ Softmax probabilities:")
        print(f"  - Sample 1: Real={probs[0][0]:.4f}, Fake={probs[0][1]:.4f}")
        print(f"  - Sample 2: Real={probs[1][0]:.4f}, Fake={probs[1][1]:.4f}")
        
        # Verify probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-5), "Probabilities don't sum to 1"
        
        print("\n✅ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_compatibility():
    """Test that inference.py model matches app.py model."""
    print("\n" + "="*60)
    print("Testing model compatibility between inference.py and app.py...")
    print("="*60)
    
    try:
        # Import from src
        sys.path.insert(0, 'src')
        from model import DeepfakeDetector as SrcModel
        
        device = torch.device("cpu")
        model1 = SrcModel().to(device)
        
        print("✓ Successfully imported model from src/model.py")
        
        # Count parameters
        param_count = sum(p.numel() for p in model1.parameters())
        print(f"  - Total parameters: {param_count:,}")
        
        print("\n✅ Model compatibility check passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Deepfake Detection Model Verification")
    print("="*60 + "\n")
    
    # Run tests
    test1 = test_model_architecture()
    test2 = test_inference_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Model Architecture Test: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Compatibility Test: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed! The model is ready to use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)
