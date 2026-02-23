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
    
    def load_model_config(self, model_name: str = 'base') -> Dict[str, Any]:
        """
        Load model configuration.
        
        Args:
            model_name: Model name (e.g., 'last_base', 'last_small')
            
        Returns:
            Model config dict
        """
        # Handle short names:
        #   'base'    → 'last_base'    → last_base.yaml   (v2 models)
        #   'base_e'  → 'last_e_base'  → last_e_base.yaml (LAST-E models)
        if not model_name.startswith('last_'):
            if model_name.endswith('_e'):
                variant = model_name[:-2]               # 'base_e' → 'base'
                model_name = f'last_e_{variant}'        # → 'last_e_base'
            else:
                model_name = f'last_{model_name}'       # → 'last_base'
            
        model_path = os.path.join(self.config_dir, 'model', f'{model_name}.yaml')
        
        if not os.path.exists(model_path):
            # Fallback or empty if not found? better to warn/error if expected.
            # For now, return empty or raise error? 
            # Given user has the file open, we must support it.
            raise FileNotFoundError(f"Model config not found: {model_path}")
        
        return self.load_yaml(model_path)

    def load_full_config(self, env_name: str = 'local', dataset_name: str = 'ntu60', model_name: str = 'base') -> Dict[str, Any]:
        """
        Load and merge environment, data, and model configs.
        
        Args:
            env_name: Environment name
            dataset_name: Dataset name
            model_name: Model variant name
            
        Returns:
            Merged config dict
        """
        env_config = self.load_environment_config(env_name)
        data_config = self.load_data_config(dataset_name)
        try:
            model_config = self.load_model_config(model_name)
        except FileNotFoundError:
            model_config = {}
        
        # Merge configs
        config = {
            'environment': env_config,
            'data': data_config,
            'model': model_config.get('model', {}) # Unwrap 'model' key if present
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


def set_nested_key(cfg: dict, key_path: str, raw_value: str) -> None:
    """
    Navigate a nested config dict via dot-notation and set the leaf to a cast value.

    Auto-casting rules (applied in order):
      'true' / 'false'        → bool
      comma-separated numbers → list[int | float]
      pure integer string     → int
      numeric string (float)  → float
      anything else           → str

    Examples:
        set_nested_key(cfg, 'training.lr', '0.1')          → cfg['training']['lr'] = 0.1
        set_nested_key(cfg, 'training.milestones', '50,65') → cfg['training']['milestones'] = [50, 65]
        set_nested_key(cfg, 'training.use_amp', 'true')    → cfg['training']['use_amp'] = True
        set_nested_key(cfg, 'model.dropout', '0.5')        → cfg['model']['dropout'] = 0.5
    """
    def _cast(s: str):
        s = s.strip()
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        if ',' in s:
            return [_cast(x) for x in s.split(',')]
        try:
            int_val = int(s)
            return int_val
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    keys = key_path.split('.')
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = _cast(raw_value)


def load_config(env: str = None, dataset: str = 'ntu60', model: str = 'base') -> Dict[str, Any]:
    """
    Convenience function to load config.
    
    Args:
        env: Environment name ('local', 'kaggle', or None for auto-detect)
        dataset: Dataset name ('ntu60' or 'ntu120')
        model: Model Name ('base', 'small', 'large')
        
    Returns:
        Config dict
    """
    loader = ConfigLoader()
    
    if env is None:
        env = loader.detect_environment()
    
    return loader.load_full_config(env, dataset, model)
