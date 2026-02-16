"""
Configuration loader utility.

Loads YAML configs and merges environment + data configs.
"""

import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configurations."""
    
    def __init__(self, config_dir: str = None):
        """
        Args:
            config_dir: Root config directory. Defaults to configs/ in project root.
        """
        if config_dir is None:
            # Get project root (3 levels up from this file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_dir = os.path.join(project_root, 'configs')
        
        self.config_dir = config_dir
    
    def load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load a YAML file."""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_environment_config(self, env_name: str = 'local') -> Dict[str, Any]:
        """
        Load environment configuration.
        
        Args:
            env_name: Environment name ('local' or 'kaggle')
            
        Returns:
            Environment config dict
        """
        env_path = os.path.join(self.config_dir, 'environment', f'{env_name}.yaml')
        
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Environment config not found: {env_path}")
        
        return self.load_yaml(env_path)
    
    def load_data_config(self, dataset_name: str = 'ntu60') -> Dict[str, Any]:
        """
        Load data configuration.
        
        Args:
            dataset_name: Dataset config name (e.g., 'ntu60', 'ntu120')
            
        Returns:
            Data config dict
        """
        data_path = os.path.join(self.config_dir, 'data', f'{dataset_name}.yaml')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data config not found: {data_path}")
        
        return self.load_yaml(data_path)
    
    def load_full_config(self, env_name: str = 'local', dataset_name: str = 'ntu60') -> Dict[str, Any]:
        """
        Load and merge environment and data configs.
        
        Args:
            env_name: Environment name
            dataset_name: Dataset name
            
        Returns:
            Merged config dict
        """
        env_config = self.load_environment_config(env_name)
        data_config = self.load_data_config(dataset_name)
        
        # Merge configs
        config = {
            'environment': env_config,
            'data': data_config
        }
        
        return config
    
    @staticmethod
    def detect_environment() -> str:
        """
        Auto-detect environment (local or Kaggle).
        
        Returns:
            'kaggle' if running on Kaggle, else 'local'
        """
        # Check for Kaggle-specific paths
        if os.path.exists('/kaggle'):
            return 'kaggle'
        return 'local'


def load_config(env: str = None, dataset: str = 'ntu60') -> Dict[str, Any]:
    """
    Convenience function to load config.
    
    Args:
        env: Environment name ('local', 'kaggle', or None for auto-detect)
        dataset: Dataset name ('ntu60' or 'ntu120')
        
    Returns:
        Config dict
    """
    loader = ConfigLoader()
    
    if env is None:
        env = loader.detect_environment()
    
    return loader.load_full_config(env, dataset)
