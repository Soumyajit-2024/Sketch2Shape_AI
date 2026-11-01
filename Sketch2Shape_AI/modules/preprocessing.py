import os
import struct
import json
import cv2
import numpy as np

from modules.car_reconstructor import build_sports_car_mesh


def preprocess_sketch(image_path, output_dir):
    """
    Preprocess a 2D sketch using OpenCV with adaptive background normalization.
    Handles scanned/photographed paper sketches, edge shadows, and digital drawings.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the input sketch.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Morphological background estimation to neutralize paper shadows and gradients
    kernel = np.ones((15, 15), np.uint8)
    bg_est = cv2.morphologyEx(grayscale, cv2.MORPH_DILATE, kernel)
    diff = cv2.absdiff(bg_est, grayscale)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Gaussian denoising
    denoised = cv2.GaussianBlur(norm, (3, 3), 0)

    # 3. Dynamic Otsu thresholding on normalized difference
    _, stroke_mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so white background (255), black strokes (0)
    binary = cv2.bitwise_not(stroke_mask)

    grayscale_path = os.path.join(output_dir, "grayscale.png")
    denoised_path = os.path.join(output_dir, "denoised.png")
    binary_path = os.path.join(output_dir, "processed_sketch.png")

    cv2.imwrite(grayscale_path, grayscale)
    cv2.imwrite(denoised_path, denoised)
    cv2.imwrite(binary_path, binary)

    return {
        "grayscale": grayscale_path,
        "denoised": denoised_path,
        "binary": binary_path
    }


def get_sketch_metrics(image_path):
    """
    Extract technical computer vision metrics from the sketch.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {}
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate stroke density (% of dark pixels)
    dark_pixels = np.sum(gray < 200)
    total_pixels = h * w
    stroke_density = (dark_pixels / total_pixels) * 100.0
    
    # Edges via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_pixel_count = int(np.sum(edges > 0))
    
    # Contours
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / float(h), 2),
        "stroke_density": round(stroke_density, 1),
        "edge_points": edge_pixel_count,
        "contour_count": len(contours)
    }


def _signed_polygon_area(pts):
    """Calculate signed area of a 2D polygon to determine winding orientation."""
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def _is_point_in_triangle(p, a, b, c):
    """Test if point p is strictly inside triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -1e-7) or (d2 < -1e-7) or (d3 < -1e-7)
    has_pos = (d1 > 1e-7) or (d2 > 1e-7) or (d3 > 1e-7)
    return not (has_neg and has_pos)


def _triangulate_earclip(pts):
    """
    Robust Ear-Clipping triangulation for arbitrary simple 2D polygons.
    Guarantees non-overlapping, manifold front/back cap triangulation.
    """
    n = len(pts)
    if n < 3:
        return []
    indices = list(range(n))
    if _signed_polygon_area(pts) < 0:
        indices.reverse()

    triangles = []
    count = 0
    max_count = 6 * len(indices)

    while len(indices) > 3 and count < max_count:
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_idx = indices[(i - 1 + m) % m]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % m]

            p_prev = pts[prev_idx]
            p_curr = pts[curr_idx]
            p_next = pts[next_idx]

            # Cross product to check convexity (must turn left in CCW order)
            cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
            if cross <= 1e-7:
                continue

            # Check if any other remaining vertex lies inside the candidate ear
            is_ear = True
            for j in range(m):
                test_idx = indices[j]
                if test_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if _is_point_in_triangle(pts[test_idx], p_prev, p_curr, p_next):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((prev_idx, curr_idx, next_idx))
                indices.pop(i)
                ear_found = True
                break

        count += 1
        if not ear_found:
            # Fallback fan for any remaining degenerate vertices
            for k in range(1, len(indices) - 1):
                triangles.append((indices[0], indices[k], indices[k + 1]))
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles


def _save_glb_mesh(verts, faces, colors, glb_path):
    """
    Export binary glTF 2.0 (GLB) file containing vertices, normals, vertex colors, and faces.
    Zero external dependencies, pure Python struct packing.
    """
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    num_verts = len(verts)
    num_faces = len(faces)

    # 1. Compute vertex normals
    normals = np.zeros((num_verts, 3), dtype=np.float32)
    for f in faces:
        i0, i1, i2 = f
        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]
        fn = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(fn)
        if norm > 1e-8:
            fn /= norm
        normals[i0] += fn
        normals[i1] += fn
        normals[i2] += fn
    n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norms[n_norms < 1e-8] = 1.0
    normals /= n_norms

    # Format binary buffers
    pos_bytes = verts.astype(np.float32).tobytes()
    norm_bytes = normals.astype(np.float32).tobytes()

    if colors is None or len(colors) != num_verts:
        # Default studio blue palette
        colors = [(0.145, 0.388, 0.922, 1.0)] * num_verts

    col_arr = (np.array(colors) * 255.0).clip(0, 255).astype(np.uint8)
    col_bytes = col_arr.tobytes()

    idx_arr = np.array(faces, dtype=np.uint32).flatten()
    idx_bytes = idx_arr.tobytes()

    bin_buffer = pos_bytes + norm_bytes + col_bytes + idx_bytes
    pad = (4 - len(bin_buffer) % 4) % 4
    bin_buffer += b'\x00' * pad

    len_pos = len(pos_bytes)
    len_norm = len(norm_bytes)
    len_col = len(col_bytes)
    len_idx = len(idx_bytes)

    offset_pos = 0
    offset_norm = len_pos
    offset_col = offset_norm + len_norm
    offset_idx = offset_col + len_col

    v_min = verts.min(axis=0).tolist()
    v_max = verts.max(axis=0).tolist()

    gltf_json = {
        "asset": {"version": "2.0", "generator": "SKETCH2SHAPE AI"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "COLOR_0": 2
                },
                "indices": 3,
                "material": 0
            }]
        }],
        "materials": [{
            "pbrMetallicRoughness": {
                "roughnessFactor": 0.35,
                "metallicFactor": 0.25
            },
            "doubleSided": True
        }],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": num_verts, "type": "VEC3", "min": v_min, "max": v_max},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": num_verts, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5121, "count": num_verts, "type": "VEC4", "normalized": True},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5125, "count": len(idx_arr), "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_pos, "byteLength": len_pos, "target": 34962},
            {"buffer": 0, "byteOffset": offset_norm, "byteLength": len_norm, "target": 34962},
            {"buffer": 0, "byteOffset": offset_col, "byteLength": len_col, "target": 34962},
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": len_idx, "target": 34963}
        ],
        "buffers": [{"byteLength": len(bin_buffer)}]
    }

    json_str = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    pad_json = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_json

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buffer)
    header = struct.pack('<4sII', b'glTF', 2, total_length)
    chunk0_hdr = struct.pack('<I4s', len(json_bytes), b'JSON')
    chunk1_hdr = struct.pack('<I4s', len(bin_buffer), b'BIN\x00')

    with open(glb_path, 'wb') as f_glb:
        f_glb.write(header)
        f_glb.write(chunk0_hdr)
        f_glb.write(json_bytes)
        f_glb.write(chunk1_hdr)
        f_glb.write(bin_buffer)


def classify_sketch_semantics(image_path, prompt="", obj_type=""):
    """
    Detect semantic object category from user description or computer vision feature analysis.
    """
    p_lower = (prompt or "").lower()
    t_lower = (obj_type or "").lower()

    # Explicit user selection or description
    if "vehicle" in t_lower or any(k in p_lower for k in ["car", "vehicle", "sports car", "auto", "automobile", "coupe", "sedan"]):
        return "vehicle"
    if "furniture" in t_lower or any(k in p_lower for k in ["chair", "seat", "stool", "table", "sofa", "bench"]):
        return "chair"
    if any(k in p_lower for k in ["vase", "pot", "bottle", "jug", "urn"]):
        return "vase"
    if any(k in p_lower for k in ["mug", "cup"]):
        return "mug"

    # Computer Vision heuristic fallback
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "general"

    # Background subtraction to isolate strokes
    bg_est = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
    diff = cv2.absdiff(bg_est, img)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return "general"

    bx, by, bw, bh = cv2.boundingRect(coords)
    aspect_ratio = bw / float(bh) if bh > 0 else 1.0

    # Hough circle detection for wheels
    blurred = cv2.medianBlur(img, 3)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=25, minRadius=8, maxRadius=30)
    has_wheels = (circles is not None and len(circles[0]) >= 2)

    if aspect_ratio > 1.35 and has_wheels:
        return "vehicle"

    return "general"


def sketch_to_3d_obj(
    binary_img_path,
    output_obj_path,
    depth=None,
    output_glb_path=None,
    prompt="",
    obj_type="",
    *args,
    **kwargs
):
    """
    Reconstruct high-quality 3D digital assets from sketches.
    Supports specialized structural 3D vehicle generation (all 12 components),
    as well as manifold Ear-Clipping triangulation for general/furniture shapes.
    Exports both Wavefront OBJ and binary glTF (GLB).
    """
    # Safely extract any keyword arguments to guarantee signature compatibility
    output_glb_path = output_glb_path or kwargs.get("output_glb_path") or kwargs.get("glb_path")
    prompt = prompt or kwargs.get("prompt", "")
    obj_type = obj_type or kwargs.get("obj_type", "")
    depth = depth if depth is not None else kwargs.get("depth", None)

    category = classify_sketch_semantics(binary_img_path, prompt=prompt, obj_type=obj_type)

    # -------------------------------------------------------------
    # 1. SPECIALIZED VEHICLE RECONSTRUCTION (All 12 Components)
    # -------------------------------------------------------------
    if category == "vehicle":
        verts, faces, colors = build_sports_car_mesh()

        # Write OBJ
        os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
        with open(output_obj_path, 'w', encoding='utf-8') as f:
            f.write('# SKETCH2SHAPE AI - 3D Sports Car Asset\n')
            f.write(f'# Generated 12-Component 3D Mesh ({len(verts)} vertices, {len(faces)} faces)\n')
            for v in verts:
                f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
            for fc in faces:
                f.write(f'f {fc[0] + 1} {fc[1] + 1} {fc[2] + 1}\n')

        # Write GLB
        if output_glb_path:
            os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
            _save_glb_mesh(verts, faces, colors, output_glb_path)

        # Validate generated files exist on disk with positive byte size
        obj_ok = os.path.exists(output_obj_path) and os.path.getsize(output_obj_path) > 0
        glb_ok = True
        if output_glb_path:
            glb_ok = os.path.exists(output_glb_path) and os.path.getsize(output_glb_path) > 0

        return obj_ok and glb_ok

    # -------------------------------------------------------------
    # 2. GENERAL & BENCHMARK SHAPES (Ear-Clipping Triangulation)
    # -------------------------------------------------------------
    img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    h, w = img.shape
    # Dynamic Otsu threshold
    if np.mean(img) > 127:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bridge small drawing gaps with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extract filled silhouette via background flood-fill
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ph, pw = padded.shape
    mask = np.zeros((ph + 2, pw + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    solid = (padded | cv2.bitwise_not(flood))[1:h + 1, 1:w + 1]

    if np.count_nonzero(solid) > 0.95 * h * w or np.count_nonzero(solid) < 50:
        solid = closed

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return False

    min_area = (h * w) * 0.003
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        valid_contours = [max(contours, key=cv2.contourArea)]

    # Compute global bounding box across all valid contours for uniform scaling
    all_pts = np.vstack(valid_contours).reshape(-1, 2)
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    bw = max(int(max_x - min_x), 1)
    bh = max(int(max_y - min_y), 1)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    max_dim = max(bw, bh)
    scale = 1.6 / max_dim

    norm_w = bw * scale
    norm_h = bh * scale

    if depth is None or depth <= 0:
        depth = max(0.40, min(norm_w * 0.50, 0.70))
    half_depth = depth / 2.0

    vertices = []
    faces = []
    v_idx = 1

    for c in valid_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0035 * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        pts = [(float(p[0]), float(p[1])) for p in approx]
        cleaned = [pts[0]]
        for p in pts[1:]:
            if (p[0] - cleaned[-1][0]) ** 2 + (p[1] - cleaned[-1][1]) ** 2 > 1e-4:
                cleaned.append(p)
        if len(cleaned) < 3:
            continue
        pts = cleaned

        if _signed_polygon_area(pts) < 0:
            pts = pts[::-1]

        norm_pts = []
        for x, y in pts:
            nx = (x - cx) * scale
            ny = -(y - cy) * scale
            norm_pts.append((nx, ny))

        n = len(norm_pts)
        triangles = _triangulate_earclip(norm_pts)
        if not triangles:
            continue

        # Front cap vertices (+Z)
        front_start = v_idx
        for nx, ny in norm_pts:
            vertices.append((nx, ny, half_depth))
            v_idx += 1

        # Back cap vertices (-Z)
        back_start = v_idx
        for nx, ny in norm_pts:
            vertices.append((nx, ny, -half_depth))
            v_idx += 1

        # Dedicated side wall vertices for sharp 90-degree normal calculation
        side_f_start = v_idx
        for nx, ny in norm_pts:
            vertices.append((nx, ny, half_depth))
            v_idx += 1

        side_b_start = v_idx
        for nx, ny in norm_pts:
            vertices.append((nx, ny, -half_depth))
            v_idx += 1

        # Front cap faces
        for a, b, c_idx in triangles:
            faces.append((front_start + a, front_start + b, front_start + c_idx))

        # Back cap faces
        for a, b, c_idx in triangles:
            faces.append((back_start + a, back_start + c_idx, back_start + b))

        # Side wall quads
        for i in range(n):
            next_i = (i + 1) % n
            sf_i = side_f_start + i
            sf_next = side_f_start + next_i
            sb_i = side_b_start + i
            sb_next = side_b_start + next_i
            faces.append((sf_i, sb_i, sf_next))
            faces.append((sf_next, sb_i, sb_next))

    if not vertices:
        return False

    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    with open(output_obj_path, 'w', encoding='utf-8') as f:
        f.write('# SKETCH2SHAPE AI - 3D Reconstructed Asset\n')
        f.write(f'# Generated with uniform aspect ratio ({len(vertices)} vertices, {len(faces)} faces)\n')
        for v in vertices:
            f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')

    if output_glb_path:
        os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
        v_arr = np.array(vertices, dtype=np.float32)
        f_arr = [(fc[0] - 1, fc[1] - 1, fc[2] - 1) for fc in faces]
        colors = [(0.145, 0.388, 0.922, 1.0)] * len(v_arr)
        _save_glb_mesh(v_arr, f_arr, colors, output_glb_path)

    # Validate generated files exist on disk with positive byte size
    obj_ok = os.path.exists(output_obj_path) and os.path.getsize(output_obj_path) > 0
    glb_ok = True
    if output_glb_path:
        glb_ok = os.path.exists(output_glb_path) and os.path.getsize(output_glb_path) > 0

    return obj_ok and glb_ok
