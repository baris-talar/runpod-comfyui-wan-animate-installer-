RunPod ComfyUI WAN Animate Installer

This repository provides an automated installer for setting up WAN Animate 2.2 inside an existing ComfyUI environment running on RunPod.

The goal is to make WAN Animate setups reproducible, stable, and low-friction by automating model downloads, custom node installation, and dependency pinning while avoiding common CUDA, Torch, and node conflicts.

This repository focuses on infrastructure and environment setup, not workflows or datasets.

⸻

What this repository provides

• One-shot installer script for WAN Animate 2.2
• Automatic download of required WAN Animate model files
• Installation of required ComfyUI custom nodes
• CUDA-aware PyTorch setup with safe dependency pinning
• Virtual environment isolation for ComfyUI
• Optional ONNX Runtime and SageAttention installation
• Designed specifically for RunPod GPU environments

⸻

Intended audience

This repository is intended for users who:

• Already have ComfyUI running
• Are using RunPod or similar JupyterLab-based GPU environments
• Want a clean, repeatable way to install WAN Animate without manual debugging

This is not a full ComfyUI installer and does not target Windows or local desktop setups.

⸻

Supported environment

• Linux (RunPod / JupyterLab)
• NVIDIA GPU (RTX-class recommended, 24GB+ VRAM)
• Existing ComfyUI directory with models/ and custom_nodes/ present
• Python 3.x available on the system

⸻

Installation (RunPod)
	1.	Navigate to your ComfyUI root directory
	2.	Upload the installer script:
runpod-comfyui-wan-animate-installer
	3.	Make the script executable
	4.	Run the script
	5.	Wait for installation to complete

The installer will:

• Download WAN Animate 2.2 model files
• Clone required ComfyUI custom nodes
• Set up a Python virtual environment
• Install pinned Torch, CUDA-compatible dependencies
• Install SageAttention and optional ONNX Runtime

After completion, start ComfyUI normally.

⸻

Configuration

The installer supports environment variable overrides.

Important options:

• HF_BASE
Hugging Face base URL hosting model files

• WANT_TORCH_STACK
auto – installs CUDA Torch if GPU is detected
keep – keeps existing Torch installation

• CUDA_TAG
CUDA wheel version (default: cu128)

• INSTALL_ONNXRUNTIME
Enable or disable onnxruntime-gpu installation

All defaults are chosen for stability on RunPod RTX GPUs.

⸻

Low-VRAM and stability notes

• Dependency pinning avoids common node breakage
• GGUF and fp8 models reduce memory pressure
• SageAttention improves performance on supported GPUs
• Virtual environments isolate ComfyUI from system Python

⸻

Troubleshooting

Most issues come from CUDA and Torch mismatches.

If you encounter Torch or CUDA errors:

• Verify your CUDA version
• Ensure Torch wheels match the CUDA version
• Avoid mixing system Python packages with the ComfyUI virtual environment

⸻

Scope and intent

• This repository focuses on ML infrastructure tooling
• Scripts are provided as-is
• No workflows, datasets, or model training code included
• Users are responsible for respecting model and dataset licenses

⸻

Credits

• ComfyUI and its open-source contributors
• WAN Animate authors
• PyTorch and NVIDIA CUDA ecosystems
