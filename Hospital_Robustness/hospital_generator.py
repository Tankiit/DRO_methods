import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.datasets import make_blobs
import cv2

class SyntheticMedicalDataset(Dataset):
    """
    Generate synthetic medical images with realistic hospital variations
    """
    def __init__(self, 
                 n_samples=10000,
                 n_hospitals=5, 
                 image_size=(224, 224),
                 modality='chest_xray'):
        
        self.n_samples = n_samples
        self.n_hospitals = n_hospitals
        self.image_size = image_size
        self.modality = modality
        
        # Hospital characteristics
        self.hospital_configs = self._generate_hospital_configs()
        
        # Generate base anatomy templates
        self.anatomy_templates = self._generate_anatomy_templates()
        
        # Generate patient demographics per hospital
        self.patient_demographics = self._generate_demographics()
        
    def _generate_hospital_configs(self):
        """Generate realistic hospital equipment/protocol configurations"""
        configs = {}
        
        for h in range(self.n_hospitals):
            configs[f'hospital_{h}'] = {
                # Equipment characteristics
                'scanner_vendor': np.random.choice(['GE', 'Siemens', 'Philips']),
                'detector_type': np.random.choice(['CR', 'DR', 'Film']),
                'kvp_range': (80 + 20*np.random.rand(), 120 + 20*np.random.rand()),
                'dose_factor': 0.8 + 0.4*np.random.rand(),  # Relative dose
                
                # Image processing pipeline
                'noise_profile': {
                    'gaussian_std': 0.01 + 0.03*np.random.rand(),
                    'poisson_lambda': 100 + 200*np.random.rand(),
                    'structured_noise': 0.001 + 0.004*np.random.rand()
                },
                
                # Acquisition protocols  
                'contrast_enhancement': 0.8 + 0.4*np.random.rand(),
                'spatial_resolution': (0.1 + 0.05*np.random.rand(), 0.1 + 0.05*np.random.rand()),
                'bit_depth': np.random.choice([12, 14, 16]),
                
                # Patient positioning variations
                'rotation_bias': -5 + 10*np.random.rand(),  # degrees
                'translation_bias': (-10 + 20*np.random.rand(), -10 + 20*np.random.rand()),
                
                # Demographics bias (realistic hospital variations)
                'age_shift': -5 + 10*np.random.rand(),  # years
                'gender_ratio': 0.4 + 0.2*np.random.rand(),  # female ratio
                'bmi_shift': -2 + 4*np.random.rand(),
            }
            
        return configs