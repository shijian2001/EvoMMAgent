# EvoMMAgent

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Create virtual environment

Choose your environment path and create it:

```bash
bash ./scripts/setup_env.sh ~/.virtualenvs/evommagent/dev
```

### 3. Activate environment

```bash
source scripts/activate_env.sh ~/.virtualenvs/evommagent/dev
```

### 4. Install system dependencies

Some tools require additional system libraries:

```bash
# Required for OCR tool (OpenCV/EasyOCR)
apt-get install -y libgl1-mesa-glx
```

### 5. Install package

```bash
# Basic installation
uv sync --active

# With development tools
uv sync --active --extra dev

# With vision-language (VL) support
uv sync --active --extra vl

# Install all extras
uv sync --active --all-extras
```

### 6. Configuration for video processing (VL only)

Set the video reader backend by exporting the environment variable:

```bash
# Use torchvision (default)
export FORCE_QWENVL_VIDEO_READER=torchvision

# Or use decord
export FORCE_QWENVL_VIDEO_READER=decord

# Or use torchcodec
export FORCE_QWENVL_VIDEO_READER=torchcodec
```

### 7. Add and remove dependencies

```bash
# Add a new package
uv add --active [package-name]

# Remove a package
uv remove --active [package-name]
```

### 8. Deactivate environment

```bash
deactivate
```

### 9. Switch to different environment

```bash
# Deactivate current environment
deactivate

# Activate another environment
source scripts/activate_env.sh /path/to/another/env
```

## Quick Start

```bash
# Run multimodal agent tests
python run_mm_agent.py
```