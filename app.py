import spaces
import json
import math
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import re
import time
from threading import Thread
from io import BytesIO
import uuid
import tempfile

import gradio as gr
import requests
import torch
from PIL import Image
import fitz
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoModel,
    AutoProcessor,
    TextIteratorStreamer,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from transformers.image_utils import load_image as hf_load_image

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.units import inch

# --- Constants and Model Setup ---
MAX_INPUT_TOKEN_LENGTH = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

# --- InternVL3_5-2B-MPO Preprocessing Functions ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- Model Loading ---
MODEL_ID_C = "HuggingFaceTB/SmolVLM-Instruct-250M"
processor_c = AutoProcessor.from_pretrained(MODEL_ID_C, trust_remote_code=True)
model_c = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID_C, trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation="flash_attention_2"
).to(device).eval()

MODEL_ID_G = "echo840/MonkeyOCR-pro-1.2B"
SUBFOLDER = "Recognition"
processor_g = AutoProcessor.from_pretrained(
    MODEL_ID_G, trust_remote_code=True, subfolder=SUBFOLDER
)
model_g = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_G, trust_remote_code=True, subfolder=SUBFOLDER, torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_I = "UCSC-VLAA/VLAA-Thinker-Qwen2VL-2B"
processor_i = AutoProcessor.from_pretrained(MODEL_ID_I, trust_remote_code=True)
model_i = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_I, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_X = "prithivMLmods/Megalodon-OCR-Sync-0713"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

# --- Moondream2 Model Loading ---
MODEL_ID_MD = "vikhyatk/moondream2"
REVISION_MD = "2025-06-21"
moondream = AutoModelForCausalLM.from_pretrained(
  MODEL_ID_MD,
  revision=REVISION_MD,
  trust_remote_code=True,
  torch_dtype=torch.float16,
  device_map={"": "cuda"},
)
tokenizer_md = AutoTokenizer.from_pretrained(MODEL_ID_MD, revision=REVISION_MD)

# --- Qwen2.5-VL-3B-Abliterated-Caption-it ---
MODEL_ID_N = "prithivMLmods/Qwen2.5-VL-3B-Abliterated-Caption-it"
processor_n = AutoProcessor.from_pretrained(MODEL_ID_N, trust_remote_code=True)
model_n = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_N, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

# --- LMM-R1-MGT-PerceReason ---
MODEL_ID_F = "VLM-Reasoner/LMM-R1-MGT-PerceReason"
processor_f = AutoProcessor.from_pretrained(MODEL_ID_F, trust_remote_code=True)
model_f = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_F, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

# TencentBAC/TBAC-VLR1-3B
MODEL_ID_G = "TencentBAC/TBAC-VLR1-3B"
processor_g = AutoProcessor.from_pretrained(MODEL_ID_G, trust_remote_code=True)
model_g = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_G, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

# OCRFlux-3B
MODEL_ID_V = "ChatDOC/OCRFlux-3B"
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V, trust_remote_code=True, torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_O = "HuggingFaceTB/SmolVLM-500M-Instruct"
processor_o = AutoProcessor.from_pretrained(MODEL_ID_O, trust_remote_code=True)
model_o = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID_O, trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation="flash_attention_2"
).to(device).eval()

# --- New Model: llava-onevision ---
MODEL_ID_LO = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
processor_lo = LlavaOnevisionProcessor.from_pretrained(MODEL_ID_LO)
model_lo = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID_LO,
    torch_dtype=torch.float16
).to(device).eval()

# OpenGVLab/InternVL3_5-2B-MPO ---
MODEL_ID_IV = 'OpenGVLab/InternVL3_5-2B-MPO'
model_iv = AutoModel.from_pretrained(
    MODEL_ID_IV,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto").eval()
tokenizer_iv = AutoTokenizer.from_pretrained(MODEL_ID_IV, trust_remote_code=True, use_fast=False)


# --- PDF Generation and Preview Utility Function ---
def generate_and_preview_pdf(image: Image.Image, text_content: str, font_size: int, line_spacing: float, alignment: str, image_size: str):
    """
    Generates a PDF, saves it, and then creates image previews of its pages.
    Returns the path to the PDF and a list of paths to the preview images.
    """
    if image is None or not text_content or not text_content.strip():
        raise gr.Error("Cannot generate PDF. Image or text content is missing.")

    # --- 1. Generate the PDF ---
    temp_dir = tempfile.gettempdir()
    pdf_filename = os.path.join(temp_dir, f"output_{uuid.uuid4()}.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_normal.fontSize = int(font_size)
    style_normal.leading = int(font_size) * line_spacing
    style_normal.alignment = {"Left": 0, "Center": 1, "Right": 2, "Justified": 4}[alignment]

    story = []

    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    page_width, _ = A4
    available_width = page_width - 2 * inch
    image_widths = {
        "Small": available_width * 0.3,
        "Medium": available_width * 0.6,
        "Large": available_width * 0.9,
    }
    img_width = image_widths[image_size]
    img = RLImage(img_buffer, width=img_width, height=image.height * (img_width / image.width))
    story.append(img)
    story.append(Spacer(1, 12))

    cleaned_text = re.sub(r'#+\s*', '', text_content).replace("*", "")
    text_paragraphs = cleaned_text.split('\n')
    
    for para in text_paragraphs:
        if para.strip():
            story.append(Paragraph(para, style_normal))

    doc.build(story)

    # --- 2. Render PDF pages as images for preview ---
    preview_images = []
    try:
        pdf_doc = fitz.open(pdf_filename)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            preview_img_path = os.path.join(temp_dir, f"preview_{uuid.uuid4()}_p{page_num}.png")
            pix.save(preview_img_path)
            preview_images.append(preview_img_path)
        pdf_doc.close()
    except Exception as e:
        print(f"Error generating PDF preview: {e}")
        
    return pdf_filename, preview_images


# --- Core Application Logic ---
@spaces.GPU
def process_document_stream(
    model_name: str, 
    image: Image.Image, 
    prompt_input: str, 
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """
    Main generator function that handles model inference tasks with advanced generation parameters.
    """
    if image is None:
        yield "Please upload an image.", ""
        return
    if not prompt_input or not prompt_input.strip():
        yield "Please enter a prompt.", ""
        return

    # --- Special Handling for Moondream2 ---
    if model_name == "Moondream2(vision)":
        image_embeds = moondream.encode_image(image)
        answer = moondream.answer_question(
            image_embeds=image_embeds,
            question=prompt_input,
            tokenizer=tokenizer_md
        )
        yield answer, answer
        return
    
    # --- Special Handling for InternVL ---
    if model_name == "OpenGVLab/InternVL3_5-2B-MPO":
        pixel_values = load_image_internvl(image, max_num=12).to(torch.bfloat16).to(device)
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        question = f"<image>\n{prompt_input}"
        response = model_iv.chat(tokenizer_iv, pixel_values, question, generation_config)
        yield response, response
        return


    processor = None
    model = None
    
    # --- Special Handling for Llava-OneVision ---
    if model_name == "llava-onevision-qwen2-0.5b-ov-hf(mini)":
        processor, model = processor_lo, model_lo
        prompt = f"<|im_start|>user <image>\n{prompt_input}<|im_end|><|im_start|>assistant"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    # --- Generic Handling for all other models ---
    else:
        if model_name == "SmolVLM-Instruct-250M(smol)": processor, model = processor_c, model_c
        elif model_name == "MonkeyOCR-pro-1.2B(ocr)": processor, model = processor_g, model_g
        elif model_name == "VLAA-Thinker-Qwen2VL-2B(reason)": processor, model = processor_i, model_i
        elif model_name == "Megalodon-OCR-Sync-0713(ocr)": processor, model = processor_x, model_x
        elif model_name == "Qwen2.5-VL-3B-Abliterated-Caption-it(caption)": processor, model = processor_n, model_n
        elif model_name == "LMM-R1-MGT-PerceReason(reason)": processor, model = processor_f, model_f 
        elif model_name == "TBAC-VLR1-3B(open-r1)": processor, model = processor_g, model_g
        elif model_name == "OCRFlux-3B(ocr)": processor, model = processor_v, model_v
        elif model_name == "SmolVLM-500M-Instruct(smol)": processor, model = processor_o, model_o
        else:
            yield "Invalid model selected.", ""
            return
            
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_input}]}]
        prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH).to(device)
    
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs, 
        "streamer": streamer, 
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": True
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer , buffer

    yield buffer, buffer


# --- Gradio UI Definition ---
def create_gradio_interface():
    """Builds and returns the Gradio web interface."""
    css = """
    .main-container { max-width: 1400px; margin: 0 auto; }
    .process-button { border: none !important; color: white !important; font-weight: bold !important; background-color: blue !important;}
    .process-button:hover { background-color: darkblue !important; transform: translateY(-2px) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }
    #gallery { min-height: 400px; }
    """
    with gr.Blocks(theme="bethecloud/storj_theme", css=css) as demo:
        gr.HTML("""
        <div class="title" style="text-align: center">
            <h1>Tiny VLMs Lab🧪</h1>
            <p style="font-size: 1.1em; color: #6b7280; margin-bottom: 0.6em;">
                Tiny VLMs for Image Content Extraction and Understanding
            </p>
        </div>
        """)

        with gr.Row():
            # Left Column (Inputs)
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=["SmolVLM-Instruct-250M(smol)", "Moondream2(vision)",
                             "OpenGVLab/InternVL3_5-2B-MPO", "Megalodon-OCR-Sync-0713(ocr)", 
                             "VLAA-Thinker-Qwen2VL-2B(reason)", "MonkeyOCR-pro-1.2B(ocr)", 
                             "Qwen2.5-VL-3B-Abliterated-Caption-it(caption)",
                             "LMM-R1-MGT-PerceReason(reason)", "OCRFlux-3B(ocr)", "TBAC-VLR1-3B(open-r1)", 
                             "SmolVLM-500M-Instruct(smol)", "llava-onevision-qwen2-0.5b-ov-hf(mini)"],
                    label="Select Model", value= "Qwen2.5-VL-3B-Abliterated-Caption-it(caption)"
                )
                
                prompt_input = gr.Textbox(label="Query Input", placeholder="✦︎ Enter the prompt")
                image_input = gr.Image(label="Upload Image", type="pil", sources=['upload'])
                
                with gr.Accordion("Advanced Settings (PDF)", open=False):
                    max_new_tokens = gr.Slider(minimum=512, maximum=8192, value=2048, step=256, label="Max New Tokens")
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                    top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                    top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                    repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
                    
                    gr.Markdown("### PDF Export Settings")
                    font_size = gr.Dropdown(choices=["8", "10", "12", "14", "16", "18"], value="12", label="Font Size")
                    line_spacing = gr.Dropdown(choices=[1.0, 1.15, 1.5, 2.0], value=1.15, label="Line Spacing")
                    alignment = gr.Dropdown(choices=["Left", "Center", "Right", "Justified"], value="Justified", label="Text Alignment")
                    image_size = gr.Dropdown(choices=["Small", "Medium", "Large"], value="Medium", label="Image Size in PDF")

                process_btn = gr.Button("🚀 Process Image", variant="primary", elem_classes=["process-button"], size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

            # Right Column (Outputs)
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.Tab("📝 Extracted Content"):
                        raw_output_stream = gr.Textbox(label="Raw Model Output Stream", interactive=False, lines=15, show_copy_button=True)
                        with gr.Row():
                            examples = gr.Examples(
                                examples=["examples/1.png", "examples/2.png", "examples/3.png",
                                          "examples/4.png", "examples/5.png", "examples/6.png"],
                                inputs=image_input, label="Examples"
                            )
                        gr.Markdown("[Report-Bug💻](https://huggingface.co/spaces/prithivMLmods/Tiny-VLMs-Lab/discussions) | [prithivMLmods🤗](https://huggingface.co/prithivMLmods)")
                    
                    with gr.Tab("📰 README.md"):
                        with gr.Accordion("(Result.md)", open=True): 
                            markdown_output = gr.Markdown()

                    with gr.Tab("📋 PDF Preview"):
                        generate_pdf_btn = gr.Button("📄 Generate PDF & Render", variant="primary")
                        pdf_output_file = gr.File(label="Download Generated PDF", interactive=False)
                        pdf_preview_gallery = gr.Gallery(label="PDF Page Preview", show_label=True, elem_id="gallery", columns=2, object_fit="contain", height="auto")

        # Event Handlers
        def clear_all_outputs():
            return None, "", "Raw output will appear here.", "", None, None

        process_btn.click(
            fn=process_document_stream,
            inputs=[model_choice, image_input, prompt_input, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=[raw_output_stream, markdown_output]
        )
        
        generate_pdf_btn.click(
            fn=generate_and_preview_pdf,
            inputs=[image_input, raw_output_stream, font_size, line_spacing, alignment, image_size],
            outputs=[pdf_output_file, pdf_preview_gallery]
        )

        clear_btn.click(
            clear_all_outputs,
            outputs=[image_input, prompt_input, raw_output_stream, markdown_output, pdf_output_file, pdf_preview_gallery]
        )
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()

    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)
