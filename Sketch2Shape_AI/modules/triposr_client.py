"""
SKETCH2SHAPE AI - TripoSR Cloud Client Module
Handles communication between the local Streamlit studio and the Google Colab TripoSR backend.
Supports health checks, live latency measurements, URL validation, and 3D asset downloading (GLB and OBJ).
"""

import os
import time
import base64
from urllib.parse import urlparse
import requests


def normalize_url(url: str) -> str:
    """Clean and normalize backend URL."""
    if not url:
        return ""
    clean = str(url).strip()
    if clean.endswith("/"):
        clean = clean[:-1]
    if not clean.startswith("http://") and not clean.startswith("https://"):
        clean = f"https://{clean}"
    return clean


def is_valid_backend_url(url: str) -> tuple[bool, str]:
    """
    Validate whether the backend URL is properly configured.
    Returns (is_valid: bool, diagnostic_message: str).
    """
    if not url or not str(url).strip():
        return False, "Backend URL is not configured. Please paste your Google Colab Cloudflare tunnel URL in Cloud AI Settings."

    clean = str(url).strip()
    clean_lower = clean.lower()

    # Reject placeholder strings
    placeholders = [
        "your-triposr-backend.trycloudflare.com",
        "your-triposr-backend",
        "xxxx.trycloudflare.com",
        "placeholder",
        "example.com",
        "your-tunnel-url",
        "<your-url>"
    ]
    for ph in placeholders:
        if ph in clean_lower:
            return False, f"Backend URL contains placeholder '{ph}'. Start your Google Colab GPU backend and paste the live generated .trycloudflare.com URL."

    # Validate protocol and domain
    norm = normalize_url(clean)
    parsed = urlparse(norm)
    netloc = parsed.netloc

    if not netloc:
        return False, f"Invalid URL format: '{clean}'. Must be a valid URL (e.g., https://abc-xyz.trycloudflare.com)."

    # Must contain a period or be localhost
    if "." not in netloc and not netloc.startswith("localhost"):
        return False, f"Invalid domain structure: '{netloc}'"

    return True, "URL format valid"


def check_backend_health(backend_url: str, timeout: int = 4) -> dict:
    """
    Check if the TripoSR cloud backend is reachable and retrieve GPU stats.

    Returns:
        dict: {
            "online": bool,
            "latency_ms": int or None,
            "gpu_name": str,
            "model": str,
            "device": str,
            "vram_total_gb": float,
            "vram_free_gb": float,
            "error": str or None
        }
    """
    is_valid, reason = is_valid_backend_url(backend_url)
    if not is_valid:
        return {
            "online": False,
            "error": reason,
            "latency_ms": None,
            "gpu_name": "N/A",
            "model": "TripoSR",
            "device": "N/A",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0
        }

    url = normalize_url(backend_url)
    health_endpoint = f"{url}/health"
    t_start = time.time()

    try:
        resp = requests.get(health_endpoint, timeout=timeout)
        latency = int((time.time() - t_start) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "online": True,
                "latency_ms": latency,
                "gpu_name": data.get("gpu_name", "NVIDIA Tesla T4"),
                "model": data.get("model", "TripoSR"),
                "device": data.get("device", "cuda:0"),
                "vram_total_gb": float(data.get("vram_total_gb", 14.56)),
                "vram_free_gb": float(data.get("vram_free_gb", 0.0)),
                "error": None
            }
        else:
            return {
                "online": False,
                "error": f"Server returned HTTP {resp.status_code}: {resp.text[:100]}",
                "latency_ms": latency,
                "gpu_name": "N/A",
                "model": "TripoSR",
                "device": "N/A",
                "vram_total_gb": 0.0,
                "vram_free_gb": 0.0
            }
    except requests.exceptions.Timeout:
        return {
            "online": False,
            "error": f"Connection timed out ({timeout}s). The Google Colab server might be starting up or overloaded.",
            "latency_ms": None,
            "gpu_name": "N/A",
            "model": "TripoSR",
            "device": "N/A",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0
        }
    except requests.exceptions.ConnectionError as ce:
        ce_str = str(ce)
        host = urlparse(url).netloc
        if "NameResolutionError" in ce_str or "getaddrinfo failed" in ce_str or "Failed to resolve" in ce_str:
            err_msg = f"DNS Error: Cannot resolve '{host}'. The Cloudflare tunnel is offline or expired. Check your Colab notebook."
        elif "Connection refused" in ce_str:
            err_msg = f"Connection refused by '{host}'. Verify that FastAPI server is listening on port 8000."
        else:
            clean_err = ce_str.split(':')[-1].strip() if ':' in ce_str else ce_str
            err_msg = f"Connection error: {clean_err}"

        return {
            "online": False,
            "error": err_msg,
            "latency_ms": None,
            "gpu_name": "N/A",
            "model": "TripoSR",
            "device": "N/A",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0
        }
    except Exception as ex:
        return {
            "online": False,
            "error": f"Backend check error: {str(ex)}",
            "latency_ms": None,
            "gpu_name": "N/A",
            "model": "TripoSR",
            "device": "N/A",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0
        }


def reconstruct_triposr(
    backend_url: str,
    image_path: str,
    output_glb_path: str,
    output_obj_path: str = None,
    mc_resolution: int = 256,
    bake_texture: bool = False,
    timeout: int = 90
) -> dict:
    """
    Send sketch to TripoSR backend, execute neural 3D reconstruction, and save GLB and OBJ.

    Returns:
        dict: {
            "success": bool,
            "glb_path": str or None,
            "obj_path": str or None,
            "inference_time_sec": float,
            "vertex_count": int,
            "face_count": int,
            "gpu_name": str,
            "error": str or None
        }
    """
    is_valid, reason = is_valid_backend_url(backend_url)
    if not is_valid:
        return {"success": False, "error": reason}

    url = normalize_url(backend_url)

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Input sketch file not found: {image_path}"}

    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    if output_obj_path:
        os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)

    t_start = time.time()
    host = urlparse(url).netloc

    # 1. Primary endpoint: /generate_bundle to fetch both GLB and OBJ
    bundle_endpoint = f"{url}/generate_bundle"
    should_try_fallback_endpoint = False

    try:
        with open(image_path, "rb") as f_img:
            files = {"file": (os.path.basename(image_path), f_img, "image/png")}
            data = {
                "mc_resolution": str(mc_resolution),
                "bake_texture": str(bake_texture).lower(),
                "foreground_ratio": "0.85"
            }
            resp = requests.post(bundle_endpoint, files=files, data=data, timeout=timeout)

        if resp.status_code == 200:
            result = resp.json()
            if not result.get("success", False):
                return {"success": False, "error": result.get("error", "Unknown TripoSR inference error")}

            # Decode and write GLB
            glb_data = base64.b64decode(result["glb_base64"])
            with open(output_glb_path, "wb") as f_glb:
                f_glb.write(glb_data)

            # Write OBJ text if available
            if output_obj_path and result.get("obj_text"):
                with open(output_obj_path, "w", encoding="utf-8") as f_obj:
                    f_obj.write(result["obj_text"])

            return {
                "success": True,
                "glb_path": output_glb_path,
                "obj_path": output_obj_path if (output_obj_path and os.path.exists(output_obj_path)) else None,
                "inference_time_sec": result.get("inference_time_sec", round(time.time() - t_start, 2)),
                "vertex_count": result.get("vertex_count", 0),
                "face_count": result.get("face_count", 0),
                "gpu_name": result.get("gpu_name", "NVIDIA Tesla T4"),
                "error": None
            }
        elif resp.status_code in (404, 405):
            # Endpoint missing on older server versions, try /generate
            should_try_fallback_endpoint = True
        else:
            return {
                "success": False,
                "error": f"TripoSR Server Error (HTTP {resp.status_code}): {resp.text[:150]}"
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Inference request timed out after {timeout} seconds on Google Colab GPU."
        }
    except requests.exceptions.ConnectionError as ce:
        ce_str = str(ce)
        if "NameResolutionError" in ce_str or "getaddrinfo failed" in ce_str or "Failed to resolve" in ce_str:
            return {
                "success": False,
                "error": f"Cloudflare Tunnel Offline: Cannot resolve '{host}'. Verify your Google Colab notebook is running and tunnel is active."
            }
        elif "Connection refused" in ce_str:
            return {
                "success": False,
                "error": f"Connection refused by '{host}'. Verify that FastAPI server is listening on port 8000."
            }
        else:
            clean_err = ce_str.split(':')[-1].strip() if ':' in ce_str else ce_str
            return {
                "success": False,
                "error": f"Network connection failed: {clean_err}"
            }
    except Exception as ex:
        # If not a network error, try fallback endpoint
        should_try_fallback_endpoint = True

    # 2. Binary stream fallback: /generate?format=glb
    if should_try_fallback_endpoint:
        try:
            generate_endpoint = f"{url}/generate"
            with open(image_path, "rb") as f_img:
                files = {"file": (os.path.basename(image_path), f_img, "image/png")}
                data = {
                    "format": "glb",
                    "mc_resolution": str(mc_resolution),
                    "bake_texture": str(bake_texture).lower(),
                    "foreground_ratio": "0.85"
                }
                resp = requests.post(generate_endpoint, files=files, data=data, timeout=timeout)

            if resp.status_code == 200:
                with open(output_glb_path, "wb") as f_out:
                    f_out.write(resp.content)

                inf_time = float(resp.headers.get("X-Inference-Time-Sec", round(time.time() - t_start, 2)))
                v_count = int(resp.headers.get("X-Vertex-Count", 0))
                f_count = int(resp.headers.get("X-Face-Count", 0))

                return {
                    "success": True,
                    "glb_path": output_glb_path,
                    "obj_path": output_obj_path if (output_obj_path and os.path.exists(output_obj_path)) else None,
                    "inference_time_sec": inf_time,
                    "vertex_count": v_count,
                    "face_count": f_count,
                    "gpu_name": "NVIDIA Tesla T4",
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": f"TripoSR Server Error (HTTP {resp.status_code}): {resp.text[:150]}"
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Inference request timed out after {timeout} seconds on Google Colab GPU."
            }
        except requests.exceptions.ConnectionError as ce:
            ce_str = str(ce)
            if "NameResolutionError" in ce_str or "getaddrinfo failed" in ce_str or "Failed to resolve" in ce_str:
                return {
                    "success": False,
                    "error": f"Cloudflare Tunnel Offline: Cannot resolve '{host}'. Verify your Google Colab notebook is running and tunnel is active."
                }
            elif "Connection refused" in ce_str:
                return {
                    "success": False,
                    "error": f"Connection refused by '{host}'. Verify that FastAPI server is listening on port 8000."
                }
            else:
                clean_err = ce_str.split(':')[-1].strip() if ':' in ce_str else ce_str
                return {
                    "success": False,
                    "error": f"Network connection failed: {clean_err}"
                }
        except Exception as ex:
            return {
                "success": False,
                "error": f"Failed to communicate with TripoSR backend: {str(ex)}"
            }

    return {"success": False, "error": "Reconstruction could not be completed."}
