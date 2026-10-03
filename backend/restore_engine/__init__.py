"""修复引擎: the AI models of the 修复 page, run in their own process.

Not part of the app's import graph: the app runs in its own environment
(CTranslate2 for whisper, the CPU onnxruntime for OCR) and these models need
a PyTorch build for the user's GPU — on a Blackwell card that is a CUDA 13
build whose cuDNN overwrites the CUDA 12 one whisper loads (same folder,
same file names). So the engine lives in a separate Python environment and
the app talks to it over pipes (protocol.py, worker.py); only protocol.py
is imported by the app, and it needs nothing but the standard library.
"""
