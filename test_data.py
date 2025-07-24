import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from feature_train import FlatImageDataset

def analyze_directory(data_dir='/home/tanmoy/research/data'):
    """Analyze the directory structure and print information about available data."""
    print("\nAnalyzing Directory Structure")
    print("=" * 50)
    
    # Check main directories
    imagenet_dir = os.path.join(data_dir, 'Imagenet')
    test_dir = os.path.join(imagenet_dir, 'test')
    
    print(f"\nDirectory Structure:")
    print(f"Base dir: {data_dir}")
    print(f"ImageNet dir: {imagenet_dir}")
    print(f"Test dir: {test_dir}")
    
    # Analyze test directory
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        test_images.sort()
        
        print(f"\nTest Images Analysis:")
        print(f"Total images: {len(test_images)}")
        print(f"Image types: {set(f.split('.')[-1].lower() for f in test_images)}")
        print(f"Sample filenames: {test_images[:5]}")
        
        # Analyze numbering pattern
        numbers = [int(f.split('.')[0]) for f in test_images if f.split('.')[0].isdigit()]
        if numbers:
            print(f"\nImage Numbering:")
            print(f"Min number: {min(numbers)}")
            print(f"Max number: {max(numbers)}")
            print(f"Total numbered images: {len(numbers)}")
            
            # Check if we can split into train/val
            train_size = int(len(numbers) * 0.8)  # 80% for training
            print(f"\nSuggested Split:")
            print(f"Training images: {train_size}")
            print(f"Validation images: {len(numbers) - train_size}")
            
            return {
                'test_dir': test_dir,
                'total_images': len(test_images),
                'image_files': test_images,
                'train_size': train_size,
                'val_size': len(numbers) - train_size
            }
    else:
        print(f"\nTest directory not found: {test_dir}")
        return None

def test_data_loading(data_info):
    """Test data loading with the analyzed directory structure."""
    if not data_info:
        print("No data information available.")
        return False
        
    print("\nTesting Data Loading")
    print("-" * 30)
    
    try:
        # Create transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create full dataset
        full_dataset = FlatImageDataset(
            root_dir=data_info['test_dir'],
            transform=transform
        )
        
        # Split into train/val
        train_size = data_info['train_size']
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, len(full_dataset) - train_size]
        )
        
        print(f"\nCreated datasets:")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Test loading batches
        print("\nTesting batch loading:")
        train_images, train_labels = next(iter(train_loader))
        print(f"Training batch shape: {train_images.shape}")
        
        val_images, val_labels = next(iter(val_loader))
        print(f"Validation batch shape: {val_images.shape}")
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_loader': train_loader,
            'val_loader': val_loader
        }
        
    except Exception as e:
        print(f"\nError in data loading:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # First analyze the directory
    data_info = analyze_directory()
    
    if data_info:
        # Then test data loading
        loaders = test_data_loading(data_info)
        if loaders:
            print("\nData loading test completed successfully!")
            print("You can use this directory structure for training.") 