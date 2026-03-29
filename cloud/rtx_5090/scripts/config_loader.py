#!/usr/bin/env python3
"""
Configuration loader for RTX 5090 training scripts.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and manages training configurations for RTX 5090."""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        self.config_dir = Path(config_dir)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the training configuration file."""
        filepath = self.config_dir / "training_config.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration."""
        return self.config.get('hardware', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.config.get('training', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config.get('dataset', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    def get_fine_tuning_config(self) -> Dict[str, Any]:
        """Get fine-tuning configuration."""
        return self.config.get('fine_tuning', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config


def main():
    """Example usage of ConfigLoader."""
    loader = ConfigLoader()
    
    print("RTX 5090 Training Configuration:")
    print(f"Platform: {loader.get_hardware_config().get('platform')}")
    print(f"GPU Memory: {loader.get_hardware_config().get('gpu_memory_gb')} GB")
    print(f"Base Model: {loader.get_model_config().get('base_model')}")
    print(f"Batch Size: {loader.get_training_config().get('batch_size')}")
    print(f"Dataset Size: {loader.get_dataset_config().get('size')}")


if __name__ == "__main__":
    main()
