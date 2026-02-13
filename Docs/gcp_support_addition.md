# GCP Support - Addition to Code Architecture

**This section should be inserted after Section 2.4 (Evaluation Config) in `code_architecture.md`**

---

## 2.5 Environment Configs (GCP Support - Config-Driven)

### 2.5.1 Local Environment (`configs/environment/local.yaml`)

```yaml
# Local Machine Environment Configuration
environment:
  name: "local"
type: "local"
  
  paths:
    data_root: "C:/Users/pathi/OneDrive/Desktop/LAST/data"
    checkpoint_root: "C:/Users/pathi/OneDrive/Desktop/LAST/checkpoints"
    log_root: "C:/Users/pathi/OneDrive/Desktop/LAST/logs"
    results_root: "C:/Users/pathi/OneDrive/Desktop/LAST/results"
    
  hardware:
    device: "cpu"                # or "cuda" if GPU available  
    num_workers: 4
    pin_memory: false
    
  cloud:
    enabled: false               # No GCS integration
```

### 2.5.2 GCP Environment (`configs/environment/gcp.yaml`)

```yaml
# GCP Instance Environment Configuration
environment:
  name: "gcp"
  type: "cloud"
  
  gcp:
    project_id: "research-project-454007"
    zone: "asia-east1-c"
    instance_name: "last-training-gpu"   # Auto-detected if on GCP
    
  paths:
    data_root: "/mnt/local-ssd/data"     # Local SSD for fast I/O
    checkpoint_root: "/home/pathi/last/checkpoints"
    log_root: "/home/pathi/last/logs"
    results_root: "/home/pathi/last/results"
    temp_workspace: "/tmp/last_training"
    
  cloud:
    enabled: true
    gcs_bucket: "last-research-bucket"
    
    sync:
      upload_checkpoints: true
      upload_logs: true
      upload_results: true
      download_data: true
      
      gcs_paths:
        dataset: "datasets/ntu120"
        checkpoints: "checkpoints/last"
        logs: "logs/last"
        results: "results/last"
        teacher_logits: "teacher_logits/ntu120"
    
    auto_shutdown:
      enabled: true
      on_success: "stop"             # or "delete"
      on_failure: "keep_running"
      
  hardware:
    device: "cuda"
    num_workers: 8
    pin_memory: true
    persistent_workers: true
```

### 2.5.3 GCP Instance Spec (`configs/environment/gcp_instance.yaml`)

```yaml
# GCP Instance Creation/Management
instance:
  name: "last-training-gpu"
  zone: "asia-east1-c"
  project_id: "research-project-454007"
  
  machine:
    type: "n1-standard-8"          # 8 vCPUs, 30GB RAM
    
    gpu:
      enabled: true
      type: "nvidia-tesla-t4"
      count: 1
      
    disk:
      boot:
        size_gb: 100
        type: "pd-ssd"
      data:
        local_ssd:
          enabled: true
          count: 1                 # 375 GB
          interface: "NVME"
          
  image:
    family: "pytorch-latest-gpu"
    project: "deeplearning-platform-release"
    
  preemptible: false
  
  startup_script: |
    #!/bin/bash
    # Mount local SSD
    if ! mountpoint -q /mnt/local-ssd; then
      sudo mkfs.ext4 -F /dev/nvme0n1
      sudo mkdir -p /mnt/local-ssd
      sudo mount /dev/nvme0n1 /mnt/local-ssd
      sudo chmod 777 /mnt/local-ssd
    fi
    
cost:
  n1_standard_8: 0.38
  tesla_t4: 0.35
  local_ssd_375gb: 0.08
  total_per_hour: 0.83
  training_time_hours: 40
  total_cost_estimate: 33.20
```

---

## 3.9 Cloud Module (`src/cloud/`) - NEW

### Class: EnvironmentDetector

**Purpose:** Auto-detect execution environment (local vs GCP).

```python
class EnvironmentDetector:
    """
    Automatically detect if running on local machine or GCP.
    
    Class Methods:
        is_gcp() -> bool
            Returns: True if running on GCP instance
            Implementation: Check GCP metadata server
                curl -s -f http://metadata.google.internal/computeMetadata/v1/
            
        get_instance_metadata() -> dict
            Returns: {
                'name': str,
                'zone': str,
                'machine_type': str,
                'preemptible': bool
            } or None if not on GCP
            
        load_environment_config() -> dict
            Returns: Merged config from environment-specific YAML
            Auto-selects: configs/environment/gcp.yaml or local.yaml
            
        resolve_paths(config: dict) -> dict
            Inputs: Config dictionary
            Returns: Config with resolved absolute paths
            (Handles both Windows and Linux paths automatically)
    """
```

### Class: GCSManager

**Purpose:** Google Cloud Storage operations.

```python
class GCSManager:
    """
    Manage Google Cloud Storage uploads/downloads.
    
    Inputs (constructor):
        - bucket_name: str
        - project_id: str
        - config: dict - GCS sync configuration
        
    Methods:
        upload_directory(local_dir: str, gcs_path: str) -> bool
            Inputs: Local directory, GCS destination path
            Returns: True if successful
            Implementation: gsutil -m rsync -r local_dir gs://bucket/gcs_path
            
        download_directory(gcs_path: str, local_dir: str) -> bool
            Inputs: GCS source path, local destination
            Returns: True if successful
            
        upload_file(local_file: str, gcs_path: str) -> bool
            Single file upload
            
        sync_checkpoints(checkpoint_dir: str) -> None
            Automatically upload new checkpoints to GCS
            
        sync_results(results_dir: str) -> None
            Upload training results to GCS
            
        download_dataset(gcs_dataset_path: str, local_data_root: str) -> None
            Download dataset from GCS on training start
            
        is_available() -> bool
            Returns: True if gsutil is installed and authenticated
    """
```

### Class: InstanceManager

**Purpose:** GCP instance lifecycle management.

```python
class InstanceManager:
    """
    Manage GCP instance lifecycle.
    
    Inputs (constructor):
        - config: dict - GCP instance configuration
        
    Methods:
        stop_instance(instance_name: str, zone: str) -> None
            Inputs: Instance name, zone
            Returns: None (stops the instance)
            Command: gcloud compute instances stop
            
        delete_instance(instance_name: str, zone: str) -> None
            Delete instance (use with caution!)
            
        auto_shutdown_on_completion(success: bool) -> None
            Inputs: Training success status
            Behavior: Reads config.cloud.auto_shutdown settings
            - If success=True and on_success="stop": stop instance
            - If success=False and on_failure="keep_running": do nothing
            
        get_instance_info() -> dict
            Returns: Current instance details from metadata
```

---

## 4.6 GCP Scripts (`scripts/gcp/`) - NEW

### Script: upload_to_gcp.py

**Purpose:** Upload code from local machine to GCP instance.

```python
"""
Upload pipeline code to GCP instance.

Usage:
    python scripts/gcp/upload_to_gcp.py --env gcp

Arguments:
    --env: Environment config (default: gcp)
    --instance: Override instance name
    --zone: Override zone
    --project: Override project ID

Functions:
    upload_code_to_instance(instance_config: dict) -> None
        Inputs: Instance configuration from gcp_instance.yaml
        Actions:
            1. Package local code (src/, configs/, scripts/)
            2. Use gcloud compute scp --recurse to upload
            3. Verify upload success
            
    get_instance_details() -> tuple
        Returns: (instance_name, zone, project_id)
        Sources: Config file or command line args
"""
```

### Script: download_results.py

**Purpose:** Download training results from GCS to local machine.

```python
"""
Download results from GCS bucket to local machine.

Usage:
    python scripts/gcp/download_results.py --experiment last_baseline_001

Arguments:
    --experiment: Experiment name (optional, downloads all if not specified)
    --output_dir: Local output directory (default: ./results)
    --bucket: Override GCS bucket name

Functions:
    download_from_gcs(
        bucket_name: str,
        gcs_path: str,
        local_dir: str
    ) -> None
        Downloads checkpoints, logs, and results
        
    list_experiments(bucket_name: str) -> list
        Returns: List of available experiment names in GCS
"""
```

### Script: setup_environment.sh

**Purpose:** Set up GCP VM environment (runs on GCP instance).

```bash
#!/bin/bash
# GCP Environment Setup Script
# Runs on the GCP instance to prepare training environment

# 1. Update system packages
sudo apt-get update

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install gcloud SDK utilities (if not present)
if ! command -v gsutil &> /dev/null; then
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi

# 4. Authenticate with GCS
gcloud auth application-default login

# 5. Mount local SSD (if not mounted)
if ! mountpoint -q /mnt/local-ssd; then
    sudo mkfs.ext4 -F /dev/nvme0n1
    sudo mkdir -p /mnt/local-ssd
    sudo mount /dev/nvme0n1 /mnt/local-ssd
    sudo chmod 777 /mnt/local-ssd
fi

# 6. Create workspace directories
mkdir -p /home/pathi/last/{checkpoints,logs,results}

# 7. Download dataset from GCS (if configured)
python -c "
from src.cloud.gcs_manager import GCSManager
from src.utils.config import ConfigLoader

config = ConfigLoader().load('configs/environment/gcp.yaml')
if config['cloud']['sync']['download_data']:
    gcs = GCSManager(config['cloud']['gcs_bucket'], config['gcp']['project_id'], config)
    gcs.download_dataset(
        config['cloud']['sync']['gcs_paths']['dataset'],
        config['paths']['data_root']
    )
"

echo "Environment setup complete!"
```

### Script: run_training.sh

**Purpose:** Main GCP training orchestrator.

```bash
#!/bin/bash
# GCP Training Orchestrator
# Handles full training pipeline with auto-shutdown

set -e  # Exit on error

# Detect environment
export LAST_ENV="gcp"

# Run training
python scripts/train.py \
    --env gcp \
    --data_config configs/data/ntu120_xsub.yaml \
    --model_config configs/model/last_base.yaml \
    --train_config configs/train/baseline.yaml 

# Check exit status
TRAIN_STATUS=$?

# Sync results to GCS
python -c "
from src.cloud.gcs_manager import GCSManager
from src.utils.config import ConfigLoader

config = ConfigLoader().load('configs/environment/gcp.yaml')
gcs = GCSManager(config['cloud']['gcs_bucket'], config['gcp']['project_id'], config)

# Upload checkpoints, logs, results
gcs.sync_checkpoints(config['paths']['checkpoint_root'])
gcs.upload_directory(config['paths']['log_root'], config['cloud']['sync']['gcs_paths']['logs'])
gcs.upload_directory(config['paths']['results_root'], config['cloud']['sync']['gcs_paths']['results'])
"

# Auto-shutdown logic
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "Training completed successfully!"
    python -c "
from src.cloud.instance_manager import InstanceManager
from src.utils.config import ConfigLoader

config = Config Loader().load('configs/environment/gcp.yaml')
manager = InstanceManager(config)
manager.auto_shutdown_on_completion(success=True)
"
else
    echo "Training failed!"
    python -c "
from src.cloud.instance_manager import InstanceManager
from src.utils.config import ConfigLoader

config = ConfigLoader().load('configs/environment/gcp.yaml')
manager = InstanceManager(config)
manager.auto_shutdown_on_completion(success=False)
"
fi
```

---

## 10. GCP Workflow - Complete Example

### Local Machine → GCP Training → Results Download

**Step 1: Prepare Code Locally**
```bash
# On your laptop (Windows)
cd C:\Users\pathi\OneDrive\Desktop\LAST

# Ensure configs are set
# configs/environment/gcp.yaml (GCP settings)
# configs/environment/gcp_instance.yaml (VM specs)
```

**Step 2: Upload to GCP**
```bash
python scripts/gcp/upload_to_gcp.py
```

What this does:
- Reads `configs/environment/gcp_instance.yaml`
- Uses `gcloud compute scp --recurse` to upload:
  - `src/`
  - `configs/`
  - `scripts/`
  - `requirements.txt`
- Uploads to: `instance_name:~/last/`

**Step 3: SSH and Run (Automatic)**
```bash
# The upload script will print:
gcloud compute ssh last-training-gpu --zone=asia-east1-c

# On GCP instance:
cd ~/last
bash scripts/gcp/setup_environment.sh
bash scripts/gcp/run_training.sh
```

**Step 4: Training Runs (Automatic)**
- Dataset downloaded from GCS (if enabled)
- Training executes on GPU
- Checkpoints saved locally  
- Logs streamed to local files
- **Auto-sync to GCS** every N epochs

**Step 5: Completion (Automatic)**
- Final results uploaded to GCS
- Instance **auto-stops** (if configured)

**Step 6: Download Results (Local Machine)**
```bash
# Back on your laptop
python scripts/gcp/download_results.py --experiment last_baseline_001
```

Downloaded to: `C:\Users\pathi\OneDrive\Desktop\LAST\results\last_baseline_001\`

---

## 11. Key Differences: Local vs GCP (Config-Driven!)

| Aspect | Local | GCP |
|--------|-------|-----|
| **Config File** | `configs/environment/local.yaml` | `configs/environment/gcp.yaml` |
| **Auto-Detection** | No metadata API | `curl metadata.google.internal` |
| **Data Path** | `C:/Users/pathi/.../data` | `/mnt/local-ssd/data` |
| **Checkpoint Sync** | Local only | Auto-upload to GCS |
| **Auto-Shutdown** | No | Optional (config-driven) |
| **Training Command** | `python scripts/train.py --env local ...` | `bash scripts/gcp/run_training.sh` |

**Critical Point:** The **only** change needed is the `--env` flag!  
```bash
# Local
python scripts/train.py --env local ...

# GCP
python scripts/train.py --env gcp ...
```

All paths, hardware settings, GCS sync behavior are **automatically resolved** from the environment config!

---

**This addition ensures:** Zero code changes between local and GCP execution. Everything is config-driven!
