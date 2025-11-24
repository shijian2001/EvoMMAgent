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

### 4. Install package

```bash
# Basic installation
uv sync --active

# With development tools
uv sync --active --extra dev
```

### 5. Add and remove dependencies

```bash
# Add a new package
uv add --active [package-name]

# Remove a package
uv remove --active [package-name]
```

### 6. Deactivate environment

```bash
deactivate
```

### 7. Switch to different environment

```bash
# Deactivate current environment
deactivate

# Activate another environment
source scripts/activate_env.sh /path/to/another/env
```

## Quick Start

```bash
# Run the example test
python run_test.py
```