# ALOE — GCP NVIDIA L4 Cloud GPU Execution Guide
**Version:** 1.0.0 · **Date:** 2026-09-18 · **Author:** Raqeeb Rameez (IIT ID: 20230934 / UoW ID: w2084794)  
**Status:** Active Execution Blueprint · **Scope:** Falcon-H1 1.5B Mamba-2 Offloading

---

## 1. Executive Summary & Architectural Motivation

ALOE's core thesis requires evaluating Parameter-Efficient Fine-Tuning (PEFT) across heterogeneous model architectures on agentic tool-use tasks (xLAM). While `google/gemma-4-E2B-it` (Standard Attention + FFN) executes efficiently on Apple Silicon MPS (~713 ms/step on an Apple M1 Max 64GB), `tiiuae/Falcon-H1-1.5B-Instruct` incorporates **7 Mamba-2 Structured State Space (SSM)** layers.

### The Apple Silicon MPS Bottleneck
On Apple Silicon, PyTorch's MPS backend lacks compiled chunked-scan CUDA/Metal kernels for Mamba-2 State Space Duality (SSD). As documented in empirical benchmarks (Gate 5 / ADR-012):
- **MPS Step Latency:** **6,918 ms/step** (a ~10× penalty compared to Gemma).
- **Campaign Projections:** Running a standard 30-trial search campaign on local MPS requires **~29 hours** of continuous, single-threaded workstation lockup.
- **Root Cause:** Sequential fallback execution of state recurrence across token chunks rather than hardware-parallelized chunked prefix-scans.

### The NVIDIA L4 Solution
By provisioning a Google Cloud Platform (GCP) **NVIDIA L4 Tensor Core GPU (24GB VRAM)** using available startup credits ($1,636 balance), we offload heavy Falcon-H1 training and multi-seed Bayesian search campaigns:
- **Compiled SSD Kernels:** Utilizing `causal-conv1d` and `mamba-ssm` compiled CUDA kernels (sm_89 Ada Lovelace architecture), chunked scan latency drops to **~280–350 ms/step** (**15×–20× acceleration** over MPS).
- **VRAM Headroom:** 24 GB GDDR6 VRAM accommodates unquantized `bfloat16` Falcon-H1 weights (~3.64 GB), optimizer states (PagedAdamW/AdamW ~7.3 GB), adapter activations, and batch sequence buffers with >50% headroom.
- **Cost-Efficiency:** Running on **Spot instances** costs **~$0.28/hour**, consuming **<$35 total** (<2.2% of startup credits) for extensive multi-seed 50-trial campaigns.

---

## 2. Hardware Tier Specifications (`g2-standard-4`)

| Hardware Component | Specification | Operational Relevance to ALOE |
| :--- | :--- | :--- |
| **Instance Type** | `g2-standard-4` | Optimized for GPU inference and single-card fine-tuning. |
| **GPU Accelerator** | 1× **NVIDIA L4** (24 GB GDDR6) | Ada Lovelace (Compute Capability `sm_89`), 4th-Gen Tensor Cores. |
| **Memory Bandwidth** | 300 GB/s with ECC | High memory bandwidth for SSM recurrent state expansion. |
| **Tensor Support** | FP8, BF16, FP16, TF32, INT8 | Full native support for unquantized `bfloat16` and FP8 acceleration. |
| **Host vCPUs** | 4 vCPUs (Intel Sapphire/Emerald Rapids) | Handles data tokenization, chat template formatting, and IO pipelines. |
| **Host System RAM** | 16 GB DDR5 | Ample headroom for xLAM dataset caching and model weights transfer. |
| **Storage Disk** | 100 GB Balanced Persistent Disk (`pd-balanced`) | Fast IOPS for HuggingFace model cache and checkpoint serialization. |
| **Operating System** | Debian 11/12 or Ubuntu 22.04 LTS DLVM | Google Deep Learning VM with CUDA 12.x pre-integrated. |

---

## 3. Cost & Credit Analysis

The project holds **$1,636.00 in GCP Startup Credits**. The L4 offload strategy is designed to consume <2.5% of total credits, reserving >$1,600 for October's Gate D Bayesian Optimization search loops and Gate E statistical validation runs.

### 3.1 Hourly Pricing Comparison (US Regions: `us-central1` / `us-east4`)

| Resource Component | On-Demand Rate ($/hr) | Spot / Preemptible Rate ($/hr) | Spot Savings (%) |
| :--- | :---: | :---: | :---: |
| **Compute (`g2-standard-4` 4 vCPU, 16GB RAM)** | $0.1706 | $0.0518 | 69.6% |
| **Accelerator (1× NVIDIA L4 24GB)** | $0.6700 | $0.2300 | 65.7% |
| **Total Instance Hourly Compute** | **$0.8406 / hr** | **$0.2818 / hr** | **66.5%** |
| **Storage (100GB `pd-balanced`)** | $0.0139 / hr ($10.00/mo) | $0.0139 / hr ($10.00/mo) | 0% |
| **Effective Total Cost per Run Hour** | **~$0.855 / hr** | **~$0.296 / hr** | **~65.4%** |

*Note: Storage when instance is in `TERMINATED/STOPPED` state costs only $0.0139/hr (~$0.33/day). No compute or GPU charges accrue while the VM is stopped.*

### 3.2 Trial Campaign Projections & Credit Depletion

A standard ALOE search campaign evaluates **50 trials** across the architectural parameter grammar (varying ranks $r \in \{4, 8, 16\}$ and scaling factors $\alpha \in \{8, 16, 32, 64\}$ across `global_attention`, `ssm`, and `ffn` groups).

#### Per-Trial Compute Breakdown on NVIDIA L4:
- **Training (200 steps, batch size 2, seq len 512):** ~60–70 seconds.
- **Evaluation (20 tool-disjoint xLAM validation samples):** ~35–45 seconds.
- **Checkpoint Serialization & Gradient Verification:** ~15 seconds.
- **Total Duration per Trial:** **~120 seconds (2.0 minutes)**.

#### Campaign Cost Model:

| Campaign Scope | Number of Trials | Total Compute Time | On-Demand Spend | Spot Spend (~$0.28/hr) | % of $1,636 Credit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pilot Calibration** | 6 trials | 0.5 hours | $0.42 | **$0.14** | 0.009% |
| **Single-Seed Search** | 50 trials (200 steps) | 2.5 hours | $2.10 | **$0.70** | 0.043% |
| **Thorough 500-Step Search** | 50 trials (500 steps) | 6.5 hours | $5.46 | **$1.83** | 0.112% |
| **Multi-Seed Robustness** | 150 trials (3 seeds × 50) | 19.5 hours | $16.39 | **$5.50** | 0.336% |
| **Comprehensive Search Campaign (with 100-hr exploration buffer)** | **~250+ trial hours** | **100.0 hours** | $84.06 | **$28.18** | **1.72%** |

**Key Takeaway:** An extensive, highly buffered 50-trial campaign with full replication and preemption tolerance consumes **~$20 to $35 total spend**, preserving **>98% of the startup credit balance**.

---

## 4. Runaway Cost Prevention & Budget Controls

To guarantee that an unattended instance cannot deplete startup credits through runaway execution or forgotten instances, the architecture mandates three concentric layers of defense:

### Layer 1: Google Cloud Budget Alerting
A hard programmatic budget alert is configured in Cloud Billing:
1. **Target Budget:** $50.00 monthly cap on the FYP GCP project.
2. **Alert Thresholds:** 50% ($25), 80% ($40), and 100% ($50).
3. **Notification Channels:** Automated email notification to the billing account administrator.

### Layer 2: Automatic GPU Idle Watchdog Daemon
The setup script includes an automated background watchdog (`gcp_l4_setup.sh --idle-shutdown 15`):
- Queries `nvidia-smi --query-gpu=utilization.gpu` every 60 seconds.
- If GPU utilization remains below 5% for **15 consecutive minutes** (indicating trial completion, crash, or idle shell):
  1. Flushes and syncs all artifacts in `runs/` to GCS.
  2. Executes `sudo poweroff`, cleanly stopping the instance.

### Layer 3: Spot Preemption & Stopping Semantics
- Instances are provisioned with `--provisioning-model=SPOT` and `--instance-termination-action=STOP`.
- When preemption occurs or the workload halts, the VM transitions to `STOPPED`. Billing for the NVIDIA L4 GPU and vCPUs terminates immediately.

---

## 5. Step-by-Step Provisioning & Execution Blueprint

### 5.1 Step 1 — Local gcloud Setup & Environment Configuration

Ensure Google Cloud SDK is installed and authenticated on your local machine:

```bash
# 1. Authenticate with Google Cloud
gcloud auth login

# 2. Set active FYP project ID
export PROJECT_ID="aloe-fyp-project"   # Replace with your exact GCP project ID
export REGION="us-central1"
export ZONE="us-central1-a"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

# 3. Enable necessary cloud services
gcloud services enable compute.googleapis.com storage.googleapis.com
```

### 5.2 Step 2 — Create Artifact Storage Bucket

Create a dedicated Google Cloud Storage (GCS) bucket for checkpoint and metrics synchronization:

```bash
export BUCKET_NAME="aloe-checkpoints-${PROJECT_ID}"

# Create uniform-access GCS bucket in same region to eliminate data transfer fees
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access

echo "Bucket created: gs://${BUCKET_NAME}"
```

### 5.3 Step 3 — Provision NVIDIA L4 Spot VM

Execute the single-command provisioning call to launch the `g2-standard-4` Spot VM with NVIDIA L4 GPU:

```bash
gcloud compute instances create aloe-l4-worker-1 \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --image-family=common-cu124-debian-11-py310 \
    --image-project=deeplearning-platform-release \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=aloe-gpu-worker \
    --metadata=install-nvidia-driver=True
```

### 5.4 Step 4 — Configure SSH & Connect to Instance

```bash
# Generate / synchronize SSH keys
gcloud compute config-ssh

# Connect directly to the VM
gcloud compute ssh aloe-l4-worker-1 --zone="${ZONE}"
```

### 5.5 Step 5 — Clone Repository & Run Automated Setup

Once connected to the VM, clone the repository and execute the automated setup script:

```bash
# 1. Clone repository
git clone https://github.com/walapalam/fyp.git
cd fyp

# 2. Export environment variables
export ALOE_GCS_BUCKET="gs://aloe-checkpoints-aloe-fyp-project"

# 3. Execute automated L4 setup pipeline (compiles kernels, installs ALOE, runs self-test)
bash aloe/scripts/cloud/gcp_l4_setup.sh --all
```

The script will:
1. Verify the NVIDIA L4 GPU and CUDA 12.x drivers via `nvidia-smi`.
2. Provision a dedicated virtual environment at `~/.venvs/aloe-l4`.
3. Install PyTorch 2.4+ with CUDA 12.4 support.
4. Compile `causal-conv1d` with `sm_89` architectural targeting.
5. Compile `mamba-ssm` with CUDA chunked-scan State Space Duality (SSD) kernels.
6. Install ALOE in editable mode (`pip install -e ".[cuda,dev]"`).
7. Execute self-test assertions verifying GPU device recognition, VRAM, and kernel imports.

---

## 6. Headless Trial Execution & Remote Monitoring

### 6.1 Launch Headless Trial with Auto-Sync

To prevent SSH disconnection from terminating runs, execute trials inside a persistent `tmux` session or using the automated headless runner:

```bash
# Launch a headless Falcon-H1 trial (200 steps)
bash aloe/scripts/cloud/gcp_l4_setup.sh --run-trial tiiuae/Falcon-H1-1.5B-Instruct P1 200

# Arm the idle watchdog (auto-shuts down VM after 15 minutes of inactivity)
bash aloe/scripts/cloud/gcp_l4_setup.sh --idle-shutdown 15
```

### 6.2 Monitor Live Run Progress

```bash
# View real-time output log
tail -f aloe/runs/cloud/trial_*.log

# Monitor GPU utilization, VRAM allocation, and temperature
watch -n 1 nvidia-smi
```

### 6.3 Manual Checkpoint Sync to GCS

To synchronize checkpoints and trial envelopes immediately to GCS:

```bash
bash aloe/scripts/cloud/gcp_l4_setup.sh --sync "gs://${BUCKET_NAME}"
```

---

## 7. Artifact Recovery to Local Workstation & VM Lifecycle

### 7.1 Sync Artifacts from GCS to Local Mac

On your local development machine (Apple Silicon):

```bash
# Pull trial logs, envelopes, and adapter checkpoints to local repo
gcloud storage rsync -r "gs://${BUCKET_NAME}/runs/" ./aloe/runs/cloud/
```

### 7.2 Stopping vs. Deleting the Instance

To maintain zero compute costs while retaining disk state, dependencies, and compiled kernels:

```bash
# Stop VM (Halts GPU/vCPU billing; retains 100GB disk at ~$0.33/day)
gcloud compute instances stop aloe-l4-worker-1 --zone="${ZONE}"

# Restart VM when ready for next batch of trials
gcloud compute instances start aloe-l4-worker-1 --zone="${ZONE}"

# Delete VM completely when all September/October campaigns are finalized
gcloud compute instances delete aloe-l4-worker-1 --zone="${ZONE}"
```

---

## 8. Verification & Operational Checklist

Before declaring the cloud offload pipeline operational, verify the following:

- [x] **CUDA & Kernel Compilation:** `import mamba_ssm` and `import causal_conv1d_cuda` execute without errors.
- [x] **Headless Execution:** Trial continues executing after detaching SSH session.
- [x] **GCS Data Pipeline:** `runs/` directory mirrors cleanly to `gs://aloe-checkpoints-...`.
- [x] **Cost Watchdog:** Idle monitor successfully triggers ACPI shutdown when GPU utilization is zero.
- [x] **Reproducibility:** Fresh process reload test confirms saved adapters deserialize cleanly on CUDA device.
