#!/usr/bin/env python3
"""Configuration loader for 4x_b200 training scripts."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and manages training configurations for 4x_b200."""
    
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
        return self.config.get('hardware', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config.get('dataset', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        return self.config.get('output', {})
    
    def get_fine_tuning_config(self) -> Dict[str, Any]:
        return self.config.get('fine_tuning', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        return self.config.get('evaluation', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        return self.config


def main():
    loader = ConfigLoader()
    print("4X B200 Training Configuration:")
    print(f"Platform: {loader.get_hardware_config().get('platform')}")
    print(f"GPU Memory: {loader.get_hardware_config().get('gpu_memory_gb')} GB")
    print(f"Base Model: {loader.get_model_config().get('base_model')}")


if __name__ == "__main__":
    main()
