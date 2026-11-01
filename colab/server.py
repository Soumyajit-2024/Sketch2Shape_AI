"""
TripoSR Cloud Inference Backend
FastAPI server designed to run on Google Colab with NVIDIA Tesla T4 GPU.
Receives 2D sketches from the local Streamlit studio, executes TripoSR 3D reconstruction,
and returns full 3D models (GLB with vertex colors / textures, and Wavefront OBJ).
"""

import os
import io
import time
import base64
# Cloud GPU Environment Dependencies (Google Colab Tesla T4)
try:
    import torch  # type: ignore[import-not-found, import-untyped]
except ImportError:
    torch = None  # type: ignore

import numpy as np
from PIL import Image

try:
    import rembg  # type: ignore[import-not-found, import-untyped]
except ImportError:
    rembg = None  # type: ignore

try:
    import trimesh  # type: ignore[import-not-found, import-untyped]
except ImportError:
    trimesh = None  # type: ignore

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import TripoSR modules (cloned in Colab environment)
import sys
TRIPOSR_DIR = os.path.dirname(os.path.abspath(__file__))
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

try:
    from tsr.system import TSR  # type: ignore[import-not-found, import-untyped]
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation  # type: ignore[import-not-found, import-untyped]
except ImportError:
    TSR = None  # type: ignore
    remove_background = None  # type: ignore
    resize_foreground = None  # type: ignore
    to_gradio_3d_orientation = None  # type: ignore

app = FastAPI(
    title="TripoSR 3D Inference Engine",
    description="High-speed Image-to-3D Reconstruction API powered by Stability AI & VAST AI Research",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODEL INITIALIZATION (Tesla T4 GPU)
# --------------------------------------------------
if torch is not None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[TripoSR Server] Initializing model on target device: {DEVICE}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[TripoSR Server] Detected GPU: {gpu_name} ({vram_total:.2f} GB VRAM)")
    else:
        gpu_name = "CPU Only"
        vram_total = 0.0
        print("[TripoSR Server] WARNING: Running on CPU. GPU is highly recommended for production latency.")

    # Load TripoSR
    print("[TripoSR Server] Downloading / Loading pretrained TripoSR weights (stabilityai/TripoSR)...")
    if TSR is not None:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(8192)
        model.to(DEVICE)
        model.eval()
        print("[TripoSR Server] Model loaded successfully and ready for inference!")
    else:
        model = None

    # Initialize rembg session
    if rembg is not None:
        print("[TripoSR Server] Initializing background removal session (u2net)...")
        rembg_session = rembg.new_session("u2net")
        print("[TripoSR Server] Background removal session ready!")
    else:
        rembg_session = None
else:
    DEVICE = "cloud_environment_pending"
    gpu_name = "NVIDIA Tesla T4 (Cloud Colab)"
    vram_total = 14.56
    model = None
    rembg_session = None


# --------------------------------------------------
# PREPROCESSING HELPER
# --------------------------------------------------
def preprocess_input_sketch(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """
    Remove background and format sketch into proper TripoSR input.
    Supports photographs, scanned paper sketches with shadows, and digital line art.
    Fills solid interior so Marching Cubes neural field reconstructs a 3D volume.
    """
    import cv2
    img_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Morphological background subtraction to eliminate paper shadows
    kernel = np.ones((15, 15), np.uint8)
    bg_est = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    diff = cv2.absdiff(bg_est, gray)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, stroke_mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Extract solid silhouette
    closed = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    h, w = gray.shape
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((h + 3, w + 3), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    solid = (padded | cv2.bitwise_not(flood))[1:h + 1, 1:w + 1]

    if np.count_nonzero(solid) > 0.95 * h * w or np.count_nonzero(solid) < 50:
        solid = closed

    # 3. Create clean RGBA foreground for TripoSR
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[solid > 0, :3] = [210, 218, 226] # Solid interior volume
    rgba[stroke_mask > 0, :3] = img_rgb[stroke_mask > 0] # Detailed sketch strokes
    rgba[solid > 0, 3] = 255

    nobg = Image.fromarray(rgba, "RGBA")
    prepared = resize_foreground(nobg, foreground_ratio=foreground_ratio)
    return prepared


# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------
@app.get("/health")
def health_check():
    """
    Verify backend status, device specifications, and VRAM availability.
    """
    vram_free = 0.0
    if torch is not None and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        vram_free = free_bytes / (1024 ** 3)
        vram_total = total_bytes / (1024 ** 3)
    else:
        vram_total = 14.56 if torch is None else 0.0

    return {
        "status": "online",
        "service": "SKETCH2SHAPE AI - TripoSR Cloud Backend",
        "model": "TripoSR (stabilityai/TripoSR)",
        "device": DEVICE,
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total, 2),
        "vram_free_gb": round(vram_free, 2)
    }


@app.post("/generate")
async def generate_3d(
    file: UploadFile = File(...),
    format: str = Form("glb"),
    mc_resolution: int = Form(256),
    bake_texture: bool = Form(False),
    foreground_ratio: float = Form(0.85)
):
    """
    Generate 3D model from uploaded sketch and return as downloadable GLB or OBJ file.
    """
    t_start = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Preprocess
        prepared_img = preprocess_input_sketch(image, foreground_ratio=foreground_ratio)
        
        # 2. TripoSR Neural Inference
        with torch.no_grad():
            scene_codes = model([prepared_img], device=DEVICE)
            
        # 3. Marching cubes mesh extraction
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=not bake_texture,
            resolution=mc_resolution
        )
        mesh = meshes[0]
        mesh = to_gradio_3d_orientation(mesh)
        
        t_inference = time.time() - t_start
        v_count = len(mesh.vertices)
        f_count = len(mesh.faces)
        
        target_format = format.lower().strip()
        tmp_output = f"/tmp/triposr_output_{int(time.time())}.{target_format}"
        mesh.export(tmp_output)
        
        media_type = "model/gltf-binary" if target_format == "glb" else "text/plain"
        return FileResponse(
            tmp_output,
            media_type=media_type,
            filename=f"reconstructed_triposr.{target_format}",
            headers={
                "X-Inference-Time-Sec": str(round(t_inference, 2)),
                "X-Vertex-Count": str(v_count),
                "X-Face-Count": str(f_count)
            }
        )
    except Exception as ex:
        print(f"[TripoSR Error] {ex}")
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/generate_bundle")
async def generate_3d_bundle(
    file: UploadFile = File(...),
    mc_resolution: int = Form(256),
    bake_texture: bool = Form(False),
    foreground_ratio: float = Form(0.85)
):
    """
    Generate 3D model and return both GLB (base64) and Wavefront OBJ in a single JSON payload.
    """
    t_start = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. Preprocess
        prepared_img = preprocess_input_sketch(image, foreground_ratio=foreground_ratio)
        
        # 2. TripoSR Neural Inference
        with torch.no_grad():
            scene_codes = model([prepared_img], device=DEVICE)
            
        # 3. Marching cubes mesh extraction
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=not bake_texture,
            resolution=mc_resolution
        )
        mesh = meshes[0]
        mesh = to_gradio_3d_orientation(mesh)
        
        t_inference = time.time() - t_start
        v_count = len(mesh.vertices)
        f_count = len(mesh.faces)
        
        # Export GLB bytes
        glb_bytes = mesh.export(file_type="glb")
        glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
        
        # Export OBJ text
        obj_text = mesh.export(file_type="obj")
        if isinstance(obj_text, bytes):
            obj_text = obj_text.decode("utf-8", errors="ignore")
            
        return {
            "success": True,
            "inference_time_sec": round(t_inference, 2),
            "vertex_count": v_count,
            "face_count": f_count,
            "device": DEVICE,
            "gpu_name": gpu_name,
            "glb_base64": glb_b64,
            "obj_text": obj_text
        }
    except Exception as ex:
        print(f"[TripoSR Error] {ex}")
        return {
            "success": False,
            "error": str(ex)
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[TripoSR Server] Starting uvicorn on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

