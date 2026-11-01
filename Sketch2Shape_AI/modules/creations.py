"""
SKETCH2SHAPE AI - Creations Management Module
Handles local archiving and retrieval of generated 2D sketches and 3D assets (GLB and OBJ).
"""

import os
import json
import time
import shutil
from datetime import datetime
from PIL import Image

CREATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "creations"
)

os.makedirs(CREATIONS_DIR, exist_ok=True)


def save_creation(
    sketch_image,
    processed_paths=None,
    obj_path=None,
    glb_path=None,
    prompt="",
    obj_type="Object",
    material="Standard",
    style="Modern",
    title=None,
    engine="OpenCV Extrusion Fallback",
    vertex_count=0,
    face_count=0
):
    """
    Save a generated sketch and reconstructed asset (GLB and/or OBJ) to the local gallery.
    
    Returns:
        dict: The saved creation metadata
    """
    os.makedirs(CREATIONS_DIR, exist_ok=True)
    creation_id = f"creation_{int(time.time())}"
    
    if not title:
        if prompt:
            title = prompt.split('.')[0][:30].strip()
        else:
            title = f"{style} {obj_type}".strip()
            
    now = datetime.now()
    date_str = now.strftime("%b %d, %Y - %I:%M %p")
    
    # Save thumbnail
    thumb_path = os.path.join(CREATIONS_DIR, f"{creation_id}_thumb.png")
    if isinstance(sketch_image, Image.Image):
        sketch_image.save(thumb_path)
    elif isinstance(sketch_image, str) and os.path.exists(sketch_image):
        img = Image.open(sketch_image)
        img.save(thumb_path)
    else:
        thumb_path = None

    # Save OBJ
    saved_obj_path = None
    if obj_path and os.path.exists(obj_path):
        saved_obj_path = os.path.join(CREATIONS_DIR, f"{creation_id}_model.obj")
        shutil.copyfile(obj_path, saved_obj_path)

    # Save GLB
    saved_glb_path = None
    if glb_path and os.path.exists(glb_path):
        saved_glb_path = os.path.join(CREATIONS_DIR, f"{creation_id}_model.glb")
        shutil.copyfile(glb_path, saved_glb_path)

    has_3d = bool(saved_glb_path or saved_obj_path)

    metadata = {
        "id": creation_id,
        "title": title,
        "date": date_str,
        "timestamp": time.time(),
        "prompt": prompt or "No description provided",
        "object_type": obj_type,
        "material": material,
        "style": style,
        "engine": engine,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "thumb_path": thumb_path,
        "obj_path": saved_obj_path,
        "glb_path": saved_glb_path,
        "has_3d": has_3d
    }

    meta_file = os.path.join(CREATIONS_DIR, f"{creation_id}_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def get_all_creations():
    """
    Retrieve all saved creations sorted newest first.
    """
    if not os.path.exists(CREATIONS_DIR):
        return []

    creations = []
    for file in os.listdir(CREATIONS_DIR):
        if file.endswith("_meta.json"):
            meta_path = os.path.join(CREATIONS_DIR, file)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("thumb_path") and not os.path.exists(data["thumb_path"]):
                        data["thumb_path"] = None
                    if data.get("obj_path") and not os.path.exists(data["obj_path"]):
                        data["obj_path"] = None
                    if data.get("glb_path") and not os.path.exists(data["glb_path"]):
                        data["glb_path"] = None
                    data["has_3d"] = bool(data.get("obj_path") or data.get("glb_path"))
                    creations.append(data)
            except Exception:
                continue

    creations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return creations


def delete_creation(creation_id):
    """
    Delete a creation and its associated thumbnail and 3D files.
    """
    meta_path = os.path.join(CREATIONS_DIR, f"{creation_id}_meta.json")
    thumb_path = os.path.join(CREATIONS_DIR, f"{creation_id}_thumb.png")
    obj_path = os.path.join(CREATIONS_DIR, f"{creation_id}_model.obj")
    glb_path = os.path.join(CREATIONS_DIR, f"{creation_id}_model.glb")

    for p in [meta_path, thumb_path, obj_path, glb_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    return True
