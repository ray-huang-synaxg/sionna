<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploying Sionna on NVIDIA DGX Spark (ARM / GB10 Blackwell)

This guide covers everything needed to deploy this Sionna repository on a
**NVIDIA DGX Spark** — the Grace CPU + GB10 Blackwell GPU desktop system —
and to execute all tutorial notebooks.

---

## Hardware reference

| Item | Spec |
|------|------|
| GPU | NVIDIA GB10 (Blackwell, compute capability **sm_121**) |
| GPU memory | ~120 GB unified with CPU |
| CPU | 20 ARM cores (aarch64 / SBSA) |
| System CUDA | 13.0 |
| Architecture | **aarch64** — _not_ x86_64; standard wheels do not apply |

> **Key constraint:** Standard PyTorch ≤ 2.8.x targets up to sm_90 (Hopper).
> GB10 (sm_121) requires either the **NGC PyTorch container ≥ 25.10** or
> PyTorch cu130 wheels from `download.pytorch.org/whl/cu130`.
> The NGC container (`nvcr.io/nvidia/pytorch:25.11-py3`) is the cleanest path —
> it ships sm_120/compute_120 PTX that JITs to sm_121 with no extra steps.

---

## Option A — NGC Docker container (recommended)

This option requires no manual CUDA/PyTorch wheel management.

### 1. Prerequisites

```bash
# Verify Docker is installed and the NVIDIA container runtime is available
docker --version
nvidia-smi          # should show GB10 and CUDA 13.0
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 2. Pull the NGC PyTorch image

```bash
# ~30 GB; ships PyTorch 2.9+, CUDA 13.0, cuDNN, NCCL — all ARM64-native
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

> Replace `25.11` with the latest `YY.MM` tag from
> [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
> if a newer release is available.

### 3. Launch the container

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -p 8888:8888 \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash
```

> `--ipc=host` is required for PyTorch shared-memory data loading.
> `-p 8888:8888` exposes JupyterLab to the host.

### 4. Install Sionna inside the container

```bash
# Inside the container — do NOT reinstall torch (the NGC image ships the
# GB10-capable build; overwriting it with pip's default would break GPU support)

cd /workspace

# Install Sionna RT submodule first (sionna depends on sionna-rt)
pip install ext/sionna-rt/

# Install Sionna (skips torch — NGC's version is preserved)
pip install --no-build-isolation .

# Install test extras without pulling a new torch
pip install pytest pytest-xdist

# Install JupyterLab and notebook execution tools
pip install jupyterlab nbconvert nbformat ipykernel
```

### 5. Verify GPU access

```python
# Quick sanity check — run before any notebook
python3 - <<'EOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute cap:", torch.cuda.get_device_capability(0))
    x = torch.ones(4, device="cuda") * 2
    print("GPU tensor:", x)
EOF
```

Expected output:

```
PyTorch: 2.9.0a0+...nv25.11
CUDA available: True
Device: NVIDIA GB10
Compute cap: (12, 1)
GPU tensor: tensor([2., 2., 2., 2.], device='cuda:0')
```

---

## Option B — Native venv (without Docker)

Use this if you prefer to work directly on the DGX Spark host OS.

### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
    build-essential git curl llvm        # llvm required for Sionna RT CPU backend
```

### 2. Create a virtual environment

```bash
python3.11 -m venv ~/.venv/sionna
source ~/.venv/sionna/bin/activate
pip install --upgrade pip
```

### 3. Install PyTorch (ARM64 + CUDA 13 / cu130 wheels)

```bash
# cu130 wheels are aarch64-native; sm_121 runs via PTX JIT (warns but works)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install Sionna from source

```bash
cd /path/to/sionna

# Initialise the sionna-rt submodule if not already done
git submodule update --init --recursive --remote
pip install ext/sionna-rt/
pip install .

# Test extras and JupyterLab
pip install pytest pytest-xdist
pip install jupyterlab nbconvert nbformat ipykernel
```

### 5. Register the kernel with JupyterLab

```bash
python3 -m ipykernel install --user \
    --name sionna \
    --display-name "Sionna (DGX Spark)"
```

---

## Running JupyterLab

### Interactive (browser)

```bash
# From inside the container or the activated venv
jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --notebook-dir=/workspace/tutorials
```

Open `http://<dgx-spark-ip>:8888` in a browser.  
If connecting from the same machine, use `http://localhost:8888`.

---

## Executing tutorials

### All tutorials — batch script

The repository ships `tutorials/run_all_notebooks.sh` which executes every
tutorial notebook with `nbconvert` and cleans outputs.

```bash
cd /workspace/tutorials      # or /path/to/sionna/tutorials

# Uncomment the notebooks you want in run_all_notebooks.sh, then:
bash run_all_notebooks.sh -gpu 0
```

The script:
1. Iterates the notebook list in order (PHY → RT → SYS).
2. Auto-downloads pretrained weights for `Sionna_tutorial_part4`, `Neural_Receiver`, and `Superimposed_Pilots`.
3. Executes each notebook in a fresh kernel via `jupyter nbconvert --execute`.
4. Clears CUDA cache between notebooks to prevent OOM.
5. Strips `stderr` and progress-bar lines from saved outputs.

### Single notebook

```bash
cd /workspace/tutorials
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  sys/End-to-End_Example.ipynb
```

---

### SYS tutorials — recommended execution order

> `Slice_Aware_Scheduling` reuses the `SystemLevelSimulator` instance (`sls`)
> that `End-to-End_Example` leaves in its kernel. Run them in the same kernel
> session or use the batch sequence below.

```bash
cd /workspace/tutorials

SYS_NOTEBOOKS=(
  "sys/HexagonalGrid.ipynb"
  "sys/PHY_Abstraction.ipynb"
  "sys/LinkAdaptation.ipynb"
  "sys/Power_Control.ipynb"
  "sys/Scheduling.ipynb"
  "sys/End-to-End_Example.ipynb"       # defines SystemLevelSimulator (sls)
  "sys/Slice_Aware_Scheduling.ipynb"   # Chapter 5 reuses sls if in same kernel
  "sys/SYS_Meets_RT.ipynb"
)

for nb in "${SYS_NOTEBOOKS[@]}"; do
  echo "▶ Running $nb ..."
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    "$nb"
  python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  echo "✓ $nb done"
done
```

---

### PHY tutorials — full list

```bash
cd /workspace/tutorials

PHY_NOTEBOOKS=(
  "phy/Hello_World.ipynb"
  "phy/Discover_Sionna.ipynb"
  "phy/Sionna_tutorial_part1.ipynb"
  "phy/Sionna_tutorial_part2.ipynb"
  "phy/Sionna_tutorial_part3.ipynb"
  "phy/Sionna_tutorial_part4.ipynb"                     # auto-downloads weights
  "phy/5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb"
  "phy/5G_NR_PUSCH.ipynb"
  "phy/Autoencoder.ipynb"
  "phy/Bit_Interleaved_Coded_Modulation.ipynb"
  "phy/CIR_Dataset.ipynb"
  "phy/Evolution_of_FEC.ipynb"
  "phy/Introduction_to_Iterative_Detection_and_Decoding.ipynb"
  "phy/Link_Level_Simulations_with_RT.ipynb"
  "phy/MIMO_OFDM_Transmissions_over_CDL.ipynb"
  "phy/Neural_Receiver.ipynb"                           # auto-downloads weights
  "phy/OFDM_MIMO_Detection.ipynb"
  "phy/Optical_Lumped_Amplification_Channel.ipynb"
  "phy/Pulse_Shaping_Basics.ipynb"
  "phy/Realistic_Multiuser_MIMO_Simulations.ipynb"
  "phy/Simple_MIMO_Simulation.ipynb"
  "phy/Superimposed_Pilots.ipynb"                       # auto-downloads weights
  "phy/Weighted_BP_Algorithm.ipynb"
)

for nb in "${PHY_NOTEBOOKS[@]}"; do
  echo "▶ Running $nb ..."
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    "$nb"
  python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  echo "✓ $nb done"
done
```

---

## Unit tests

```bash
cd /workspace
python3 -m pytest test/unit/ -v --tb=short
```

Run only the slice-aware scheduling tests:

```bash
python3 -m pytest test/unit/sys/test_slice_scheduler.py -v
```

---

## Environment variable reference

| Variable | Purpose | Recommended value |
|----------|---------|-------------------|
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to expose | `0` (DGX Spark has one GB10) |
| `TORCH_CUDA_ARCH_LIST` | PTX/SASS targets for custom CUDA extensions | `12.1a` |
| `TRITON_PTXAS_PATH` | Triton JIT assembler path | `/usr/local/cuda/bin/ptxas` |

```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

---

## Troubleshooting

### "no kernel image available for device" / CUDA error 209

The installed PyTorch wheel does not include a kernel for sm_121.

**Fix:** Use the NGC container (`pytorch:25.10-py3` or later), or install from
the `cu130` index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### GPU tensor operations hang or OOM between notebooks

Flush the cache between heavy simulations:

```bash
python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()"
```

### `Slice_Aware_Scheduling.ipynb` Chapter 5 falls back to synthetic data

Chapter 5 tries to reuse the `sls` (`SystemLevelSimulator`) instance from the
same kernel. If the notebook is opened in a fresh kernel, `sls` is undefined
and a synthetic fallback activates automatically.

**Fix:** Run `End-to-End_Example.ipynb` **first** in the same JupyterLab kernel,
then run `Slice_Aware_Scheduling.ipynb` from Section 2 onward — Chapter 5 will
detect `sls` and run the full 273-PRB / 30 kHz SCS simulation.

### `ModuleNotFoundError: sionna.rt`

Sionna RT is a git submodule in `ext/sionna-rt/`. Initialise it, then install:

```bash
git submodule update --init --recursive --remote
pip install ext/sionna-rt/
```

### LLVM not found (Sionna RT CPU backend)

```bash
sudo apt-get install -y llvm
# Or point Dr.Jit to your LLVM shared library
export DRJIT_LIBLLVM_PATH=$(llvm-config --libdir)/libLLVM.so
```

---

## Quick-start (copy-paste summary)

```bash
# ── 1. Pull and launch the NGC container ──────────────────────────────────────
docker pull nvcr.io/nvidia/pytorch:25.11-py3

docker run --gpus all -it --rm --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":/workspace -w /workspace -p 8888:8888 \
  nvcr.io/nvidia/pytorch:25.11-py3 bash

# ── 2. Inside container: install Sionna ───────────────────────────────────────
cd /workspace
pip install ext/sionna-rt/ .
pip install jupyterlab nbconvert nbformat ipykernel

# ── 3. Verify GB10 is visible ─────────────────────────────────────────────────
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GB10

# ── 4. Launch JupyterLab ──────────────────────────────────────────────────────
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --NotebookApp.token='' --notebook-dir=tutorials
# Open http://localhost:8888 in browser

# ── 5. Or run all SYS tutorials headless ──────────────────────────────────────
cd tutorials
for nb in \
  sys/HexagonalGrid.ipynb \
  sys/PHY_Abstraction.ipynb \
  sys/LinkAdaptation.ipynb \
  sys/Power_Control.ipynb \
  sys/Scheduling.ipynb \
  sys/End-to-End_Example.ipynb \
  sys/Slice_Aware_Scheduling.ipynb \
  sys/SYS_Meets_RT.ipynb; do
    echo "▶ $nb"
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 "$nb"
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done
```
