#!/usr/bin/env python3
"""
Test script to verify CIFAR experiment setup.
This script tests the basic functionality without running full training.
"""

import sys
import torch
import torchvision
from pathlib import Path

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        # Core PyTorch
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print("✅ PyTorch imports successful")
        
        # Computer vision
        import torchvision
        import torchvision.transforms as transforms
        print("✅ Torchvision imports successful")
        
        # Data science
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas imports successful")
        
        # ML libraries
        from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
        print("✅ Scikit-learn imports successful")
        
        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Matplotlib and Seaborn imports successful")
        
        # Deep learning utilities
        import timm
        print("✅ TIMM imports successful")
        
        # Progress bars
        from tqdm import tqdm, trange
        print("✅ TQDM imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
        print(f"   Current device: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    return torch.cuda.is_available()

def test_data_loading():
    """Test basic data loading"""
    print("\nTesting data loading...")
    
    try:
        # Test CIFAR-10 loading
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Small subset for testing
        print("Loading CIFAR-10 test set...")
        test_dataset = torchvision.datasets.CIFAR10(
            root='./test_data', 
            train=False, 
            transform=transform, 
            download=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=2
        )
        
        # Test one batch
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        print(f"✅ Data loading successful")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Dataset size: {len(test_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        # Test ResNet creation with timm
        import timm
        
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        print(f"✅ ResNet-18 creation successful")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR images
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Forward pass shape: {output.shape}")
        print(f"   Expected shape: torch.Size([2, 10])")
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_our_modules():
    """Test our custom modules"""
    print("\nTesting custom modules...")
    
    try:
        # Test if our modules can be imported
        sys.path.append('.')
        
        from multi_scoring import MultiScoreOODDetector
        print("✅ MultiScoreOODDetector import successful")
        
        from feature_train import TimmFeatureExtractor, HierarchicalDROWithMultiScoring
        print("✅ Custom feature training modules import successful")
        
        # Test TimmFeatureExtractor
        model = TimmFeatureExtractor(
            model_name='resnet18',
            num_classes=10,
            pretrained=False
        )
        
        dummy_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            logits, features = model(dummy_input)
        
        print(f"   TimmFeatureExtractor output shapes:")
        print(f"     Logits: {logits.shape}")
        print(f"     Features: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom modules error: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'cifar_resnet_experiments.py',
        'run_cifar_experiments.py', 
        'run_single_cifar.py',
        'feature_train.py',
        'multi_scoring.py',
        'README_CIFAR_EXPERIMENTS.md'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    print("CIFAR ResNet Experiment Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Data Loading Test", test_data_loading),
        ("Model Creation Test", test_model_creation),
        ("Custom Modules Test", test_our_modules),
        ("File Structure Test", test_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The setup is ready for experiments.")
        print("\nYou can now run:")
        print("  python run_single_cifar.py --dataset cifar10 --model resnet18 --epochs 5")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("Install missing dependencies with:")
        print("  pip install torch torchvision timm pandas scikit-learn matplotlib seaborn tqdm")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 