
# **[Tiny VLMs Lab](https://huggingface.co/spaces/prithivMLmods/Tiny-VLMs-Lab)**

Tiny VLMs Lab is an interactive web application built with Gradio that allows users to experiment with various small-scale Vision-Language Models (VLMs) for tasks such as image description, content extraction, OCR, reasoning, and captioning. The app supports uploading images, entering custom prompts, and generating outputs in text and PDF formats. It leverages multiple open-source models from Hugging Face, providing a streamlined interface for testing and comparing model performance on visual understanding tasks.

This project is hosted on Hugging Face Spaces and is open-source on GitHub. It demonstrates the capabilities of lightweight VLMs in a user-friendly environment, with advanced generation parameters and PDF export features.

> **Hugging Face Space/App**: [https://huggingface.co/spaces/prithivMLmods/Tiny-VLMs-Lab](https://huggingface.co/spaces/prithivMLmods/Tiny-VLMs-Lab)

## Preview

|![Image 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/BdY8XsSOXh9ayZIq6DGex.png) | ![Image 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/NVaxButskWzRNnT6EXtAS.png) | ![Image 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/EGULxIawkAqKJ8btgLVjU.png)  |
|---|---|---|


## Features

- **Model Selection**: Choose from a variety of tiny VLMs, each specialized for different tasks like fast inference, OCR, reasoning, or captioning.
- **Image Upload and Processing**: Upload images and provide custom prompts for model inference.
- **Streaming Output**: Real-time generation of model responses with a text streamer for interactive feedback.
- **Advanced Generation Controls**: Customize parameters such as max new tokens, temperature, top-p, top-k, and repetition penalty.
- **PDF Generation and Preview**: Export results to PDF with configurable font size, line spacing, alignment, and image scaling. Preview PDF pages directly in the app.
- **Markdown Rendering**: View cleaned and formatted output in a dedicated README.md-style tab.
- **Example Images**: Pre-loaded examples for quick testing.
- **Clear Functionality**: Easily reset inputs and outputs.

## Supported Models

The app integrates the following models, each with a brief description of its focus:

- **LFM2-VL-450M (fast)**: A lightweight model for quick vision-language tasks.
- **LFM2-VL-1.6B (fast)**: An enhanced version for faster and more accurate processing.
- **SmolVLM-Instruct-250M (smol)**: Compact model for instruction-following on images.
- **Moondream2 (vision)**: Specialized in visual question answering and description.
- **ShotVL-3B (cinematic)**: Designed for cinematic or detailed scene understanding.
- **Megalodon-OCR-Sync-0713 (ocr)**: Focused on optical character recognition in images.
- **VLAA-Thinker-Qwen2VL-2B (reason)**: Emphasizes reasoning over visual content.
- **MonkeyOCR-pro-1.2B (ocr)**: Advanced OCR capabilities for text extraction.
- **Qwen2.5-VL-3B-Abliterated-Caption-it (caption)**: Generates detailed captions for images.
- **Nanonets-OCR-s (ocr)**: Efficient OCR for structured text recognition.
- **LMM-R1-MGT-PerceReason (reason)**: Perceptual reasoning on visual inputs.
- **OCRFlux-3B (ocr)**: High-performance OCR model.
- **TBAC-VLR1-3B (open-r1)**: Open-ended reasoning and understanding.
- **SmolVLM-500M-Instruct (smol)**: Instructable small VLM for versatile tasks.
- **llava-onevision-qwen2-0.5b-ov-hf (mini)**: Miniature model for one-vision tasks.

All models are loaded with torch.float16 precision for efficiency on GPU-enabled environments.

## Installation

To run the app locally, ensure you have Python installed (version 3.10 or higher recommended). Follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Tiny-VLMs-Lab.git
   cd Tiny-VLMs-Lab
   ```

2. **Install Pre-Requirements** (from `pre-requirements.txt`):
   ```
   pip install -r pre-requirements.txt
   ```
   This includes:
   - pip>=23.0.0

3. **Install Dependencies** (from `requirements.txt`):
   ```
   pip install -r requirements.txt
   ```
   Key dependencies include:
   - git+https://github.com/Dao-AILab/flash-attention.git
   - git+https://github.com/huggingface/transformers.git
   - git+https://github.com/huggingface/accelerate.git
   - git+https://github.com/huggingface/peft.git
   - transformers-stream-generator
   - huggingface_hub
   - albumentations
   - pyvips-binary
   - qwen-vl-utils
   - sentencepiece
   - opencv-python
   - docling-core
   - python-docx
   - torchvision
   - safetensors
   - matplotlib
   - num2words
   - reportlab
   - xformers
   - requests
   - pymupdf
   - hf_xet
   - spaces
   - pyvips
   - pillow
   - gradio
   - einops
   - torch
   - fpdf
   - timm
   - av

   Note: Some dependencies require GPU support (e.g., CUDA for torch). Ensure your environment is set up accordingly.

4. **Run the App**:
   ```
   python app.py
   ```
   The app will launch a local Gradio server. Access it via the provided URL in your browser.

## Usage

1. **Select a Model**: Choose from the dropdown menu.
2. **Upload an Image**: Use the image input field or select from examples.
3. **Enter a Prompt**: Provide a query like "Describe the image!" or a custom instruction.
4. **Adjust Settings (Optional)**: Open the advanced accordion to tweak generation parameters or PDF export options.
5. **Process**: Click "Process Image" to generate output. Results stream in the "Extracted Content" tab.
6. **View Markdown**: Switch to the "README.md" tab for formatted output.
7. **Generate PDF**: In the "PDF Preview" tab, click "Generate PDF & Render" to create and preview a downloadable PDF.
8. **Clear**: Use the "Clear All" button to reset the interface.

For best results, use high-quality images and specific prompts tailored to the selected model's strengths.

## Hardware Requirements

- **GPU Recommended**: NVIDIA GPU with at least 64GB+ VRAM for efficient model loading and inference.
- **CPU Fallback**: Works on CPU but may be slower for larger models.
- **Memory**: At least 48GB RAM; more for handling multiple models or large images.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub. For bug reports, use the discussions section on Hugging Face Spaces.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
