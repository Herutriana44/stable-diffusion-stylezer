"""
Stable Diffusion Stylezer - Flask App
Mengganti style/outfit pada gambar dengan masking area dan upload gambar outfit.
"""

import os
import io
import base64
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import torch
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["OUTPUT_FOLDER"] = Path("outputs")

app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["OUTPUT_FOLDER"].mkdir(exist_ok=True)

# Config dari .env
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-inpainting")
SD_PROMPT = os.getenv(
    "SD_PROMPT",
    "person wearing fashionable outfit, high quality, detailed, professional photography, natural lighting",
)
SD_NEGATIVE_PROMPT = os.getenv(
    "SD_NEGATIVE_PROMPT",
    "low quality, blurry, distorted, ugly, bad anatomy, deformed",
)
SD_NUM_INFERENCE_STEPS = int(os.getenv("SD_NUM_INFERENCE_STEPS", "30"))
SD_GUIDANCE_SCALE = float(os.getenv("SD_GUIDANCE_SCALE", "7.5"))
SD_STRENGTH = float(os.getenv("SD_STRENGTH", "0.85"))

# Global pipeline (lazy load)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from diffusers import StableDiffusionInpaintPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        _pipeline = _pipeline.to(device)
    return _pipeline


def composite_outfit_into_mask(image: Image.Image, outfit: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Composite outfit image ke area yang di-mask pada image.
    Outfit di-resize untuk fit di bounding box mask.
    """
    img_arr = np.array(image.convert("RGB"))
    outfit_arr = np.array(outfit.convert("RGB").resize(image.size, Image.Resampling.LANCZOS))
    mask_arr = np.array(mask.convert("L"))

    # Normalize mask: 255 = area to replace
    if mask_arr.max() <= 1:
        mask_arr = (mask_arr * 255).astype(np.uint8)
    mask_bool = mask_arr > 127

    # Blend: di area mask, gunakan outfit
    result = img_arr.copy()
    result[mask_bool] = outfit_arr[mask_bool]

    return Image.fromarray(result)


def process_style_transfer(image: Image.Image, outfit: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Proses style transfer: composite outfit ke masked area, lalu inpainting untuk blending natural.
    """
    # 1. Composite outfit ke masked area
    composited = composite_outfit_into_mask(image, outfit, mask)

    # 2. Gunakan img2img untuk refine/blend (lebih natural daripada inpainting untuk outfit transfer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = get_pipeline()

    # Pipeline inpainting: white=inpaint, black=keep

    # Pastikan size sama
    w, h = image.size
    composited = composited.resize((w, h))
    mask_resized = mask.resize((w, h)).convert("L")

    # Run inpainting - model akan refine area yang di-mask
    result = pipe(
        prompt=SD_PROMPT,
        negative_prompt=SD_NEGATIVE_PROMPT,
        image=composited,
        mask_image=mask_resized,
        num_inference_steps=SD_NUM_INFERENCE_STEPS,
        guidance_scale=SD_GUIDANCE_SCALE,
        strength=SD_STRENGTH,
    ).images[0]

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload/image", methods=["POST"])
def upload_image():
    """Upload gambar yang ingin diubah (gambar 1)."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"success": True, "image": f"data:image/png;base64,{b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/outfit", methods=["POST"])
def upload_outfit():
    """Upload gambar outfit/stylist (gambar 2)."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"success": True, "image": f"data:image/png;base64,{b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/process", methods=["POST"])
def process():
    """
    Process: terima base64 image, outfit, mask.
    Return base64 result image.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    image_b64 = data.get("image")
    outfit_b64 = data.get("outfit")
    mask_b64 = data.get("mask")

    if not all([image_b64, outfit_b64, mask_b64]):
        return jsonify({"error": "Missing image, outfit, or mask"}), 400

    try:
        # Decode base64
        def b64_to_img(s):
            if "," in s:
                s = s.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

        def b64_to_mask(s):
            if "," in s:
                s = s.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("L")

        image = b64_to_img(image_b64)
        outfit = b64_to_img(outfit_b64)
        mask = b64_to_mask(mask_b64)

        # Pastikan mask: putih (255) = area to change, hitam (0) = keep (sesuai inpainting)
        mask_arr = np.array(mask)
        if mask_arr.max() < 128:
            mask = mask.point(lambda p: 255 - p, "L")

        # Resize mask ke ukuran image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        result = process_style_transfer(image, outfit, mask)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"success": True, "image": f"data:image/png;base64,{b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
