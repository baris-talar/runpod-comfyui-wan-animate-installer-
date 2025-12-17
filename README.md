RunPod ComfyUI WAN Animate Installer

Automated installer for setting up WAN Animate 2.2 inside an existing ComfyUI environment on RunPod.

This script handles model downloads, required custom nodes, and CUDA-compatible dependency pinning to avoid common Torch, CUDA, and node conflicts.

This repository focuses on infrastructure and environment setup, not workflows or datasets.

⸻

What this provides
	•	One-shot installer for WAN Animate 2.2
	•	Automatic download of required model files
	•	Installation of required ComfyUI custom nodes
	•	CUDA-aware PyTorch setup with pinned dependencies
	•	Virtual environment isolation
	•	Optional SageAttention and ONNX Runtime support

⸻

Intended users
	•	Users with ComfyUI already running
	•	RunPod / JupyterLab GPU environments
	•	Anyone who wants a repeatable, no-debug setup for WAN Animate

Not a full ComfyUI installer.
Not intended for Windows or local desktop setups.

⸻

Requirements
	•	Linux (RunPod / JupyterLab)
	•	NVIDIA GPU (RTX-class recommended)
	•	Existing models/ and custom_nodes/ directories
	•	Python 3.x

⸻

Usage (RunPod)
	1.	Navigate to your ComfyUI root directory
	2.	Upload runpod-comfyui-wan-animate-installer
	3.	Make it executable and run it
	4.	Wait for completion

Start ComfyUI normally after install.

⸻

Configuration

The installer supports environment variable overrides:
	•	HF_BASE – Hugging Face base URL for model files
	•	WANT_TORCH_STACK – auto or keep
	•	CUDA_TAG – CUDA wheel version (default: cu128)
	•	INSTALL_ONNXRUNTIME – enable or disable ONNX Runtime

Defaults are chosen for stability on RunPod RTX GPUs.

⸻

Scope
	•	ML infrastructure tooling only
	•	No workflows, datasets, or training code
	•	Users are responsible for license compliance

⸻

Credits
	•	ComfyUI contributors
	•	WAN Animate authors
	•	PyTorch and NVIDIA CUDA ecosystems
