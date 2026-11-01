"""
SKETCH2SHAPE AI - Advanced 3D Viewer Component
Supports both Binary glTF (GLB with vertex colors & PBR textures) and Wavefront OBJ.
Powered by Three.js WebGL, OrbitControls, and GLTFLoader.
In standby mode, renders a clean isometric 3D cube illustration with rotation arrows.
"""

import os
import base64
import streamlit.components.v1 as components


def render_3d_viewer(
    model_file_path=None,
    obj_file_path=None,
    height=480,
    model_title="3D Model Preview",
    default_color="#2563EB"
):
    """
    Renders an interactive Three.js 3D viewport supporting both GLB and OBJ.
    
    Args:
        model_file_path (str, optional): Path to .glb or .obj model file.
        obj_file_path (str, optional): Legacy argument for backward compatibility.
        height (int): Viewport height in pixels.
        model_title (str): Title displayed in the viewer header.
        default_color (str): Hex color for fallback material.
    """
    active_path = model_file_path or obj_file_path
    model_format = "none"
    model_data_js = "null"

    if active_path and os.path.exists(active_path):
        ext = os.path.splitext(active_path)[1].lower()
        if ext == ".glb":
            try:
                with open(active_path, "rb") as f_glb:
                    glb_b64 = base64.b64encode(f_glb.read()).decode("ascii")
                    model_data_js = f"'{glb_b64}'"
                    model_format = "glb"
            except Exception:
                model_format = "none"
        elif ext == ".obj":
            try:
                with open(active_path, "r", encoding="utf-8", errors="ignore") as f_obj:
                    content = f_obj.read().replace("`", "\\`").replace("${", "\\${")
                    model_data_js = f"`{content}`"
                    model_format = "obj"
            except Exception:
                model_format = "none"

    has_model = (model_format in ["glb", "obj"])

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{model_title}</title>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                user-select: none;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #F8FAFC;
                color: #0F172A;
                overflow: hidden;
                width: 100vw;
                height: {height}px;
            }}
            #container {{
                position: relative;
                width: 100%;
                height: 100%;
                background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
                border: 1px solid #E2E8F0;
                border-radius: 14px;
                overflow: hidden;
                box-shadow: inset 0 1px 3px 0 rgba(0, 0, 0, 0.02);
            }}
            #canvas3d {{
                width: 100%;
                height: 100%;
                display: {"block" if has_model else "none"};
            }}
            .toolbar {{
                position: absolute;
                top: 14px;
                left: 16px;
                right: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                pointer-events: none;
                z-index: 10;
            }}
            .header-title {{
                pointer-events: auto;
                font-size: 15px;
                font-weight: 700;
                color: #0F172A;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .format-tag {{
                font-size: 10px;
                font-weight: 800;
                text-transform: uppercase;
                background: {"#EFF6FF" if model_format == "glb" else "#F0FDF4"};
                color: {"#2563EB" if model_format == "glb" else "#16A34A"};
                padding: 2px 7px;
                border-radius: 4px;
                margin-left: 6px;
            }}
            .header-actions {{
                pointer-events: auto;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .icon-btn {{
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                color: #64748B;
                border-radius: 8px;
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            .icon-btn:hover {{
                color: #0F172A;
                border-color: #CBD5E1;
                background: #F8FAFC;
            }}
            .btn-group {{
                pointer-events: auto;
                display: flex;
                gap: 6px;
                background: rgba(255, 255, 255, 0.94);
                backdrop-filter: blur(8px);
                padding: 4px;
                border-radius: 8px;
                border: 1px solid #E2E8F0;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            }}
            .tool-btn {{
                background: transparent;
                border: none;
                color: #475569;
                font-size: 11px;
                font-weight: 600;
                padding: 5px 9px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.15s ease;
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .tool-btn:hover {{
                background: #F1F5F9;
                color: #0F172A;
            }}
            .tool-btn.active {{
                background: #0F172A;
                color: #FFFFFF;
            }}
            .hud-bar {{
                position: absolute;
                bottom: 12px;
                left: 16px;
                right: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                pointer-events: none;
                z-index: 10;
            }}
            .hud-info {{
                pointer-events: auto;
                font-size: 11px;
                color: #475569;
                font-weight: 600;
                background: rgba(255, 255, 255, 0.94);
                backdrop-filter: blur(6px);
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #E2E8F0;
            }}
            .instructions {{
                font-size: 11px;
                color: #94A3B8;
                pointer-events: none;
            }}
            /* Standby State with Isometric Cube and Rotation Arrows */
            #standby {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                pointer-events: none;
                z-index: 5;
                display: {"none" if has_model else "flex"};
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            .standby-cube {{
                margin-bottom: 16px;
            }}
            .standby-text {{
                font-size: 14px;
                font-weight: 500;
                color: #64748B;
                letter-spacing: -0.01em;
            }}
        </style>
        <!-- Load Three.js, OrbitControls, and GLTFLoader -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    </head>
    <body>
        <div id="container">
            <div class="toolbar">
                <div class="header-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                        <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                        <line x1="12" y1="22.08" x2="12" y2="12"></line>
                    </svg>
                    <span>{model_title}</span>
                    {"<span class='format-tag'>" + model_format.upper() + "</span>" if has_model else ""}
                </div>

                <div class="header-actions">
                    {"<div class='btn-group'>" if has_model else ""}
                    {"<button class='tool-btn' id='btnRotate' title='Toggle Auto-Rotation'>🔄 Rotate</button>" if has_model else ""}
                    {"<button class='tool-btn' id='btnWireframe' title='Toggle Wireframe'>🕸️ Wireframe</button>" if has_model else ""}
                    {"<button class='tool-btn' id='btnColor' title='Change Shading'>🎨 Shading</button>" if has_model else ""}
                    {"<button class='tool-btn' id='btnReset' title='Reset View'>🎯 Reset</button>" if has_model else ""}
                    {"</div>" if has_model else ""}

                    <button class="icon-btn" id="btnFullscreen" title="Toggle Fullscreen">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                        </svg>
                    </button>
                    <button class="icon-btn" title="Options">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="1"></circle>
                            <circle cx="12" cy="5" r="1"></circle>
                            <circle cx="12" cy="19" r="1"></circle>
                        </svg>
                    </button>
                </div>
            </div>

            <div id="standby">
                <div class="standby-cube">
                    <svg width="140" height="140" viewBox="0 0 160 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <!-- Rotation arrow back arc -->
                        <path d="M30 100 C20 78, 45 55, 80 55 C100 55, 120 65, 132 80" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round" stroke-dasharray="4 3"/>
                        
                        <!-- Isometric Cube -->
                        <!-- Top Face -->
                        <polygon points="80,35 115,55 80,75 45,55" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" stroke-linejoin="round"/>
                        <!-- Left Face -->
                        <polygon points="45,55 80,75 80,115 45,95" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1.5" stroke-linejoin="round"/>
                        <!-- Right Face -->
                        <polygon points="80,75 115,55 115,95 80,115" fill="#CBD5E1" stroke="#94A3B8" stroke-width="1.5" stroke-linejoin="round"/>

                        <!-- Rotation arrows foreground arc -->
                        <path d="M132 82 C142 98, 115 125, 80 125 C50 125, 26 108, 22 92" stroke="#94A3B8" stroke-width="2.2" stroke-linecap="round"/>
                        <!-- Left arrowhead -->
                        <polyline points="20,99 21,91 29,94" fill="none" stroke="#94A3B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                        <!-- Right arrowhead -->
                        <polyline points="134,75 133,83 125,80" fill="none" stroke="#94A3B8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="standby-text">Your 3D model will appear here</div>
            </div>

            <div class="hud-bar" style="display: {"flex" if has_model else "none"};">
                <div class="hud-info" id="statsHud">
                    Vertices: 0 | Faces: 0
                </div>
                <div class="instructions">
                    Left Drag: Orbit • Scroll: Zoom • Right Drag: Pan
                </div>
            </div>

            <canvas id="canvas3d"></canvas>
        </div>

        <script>
            const hasModel = {"true" if has_model else "false"};

            if (hasModel && typeof THREE !== 'undefined') {{
                initWebGL();
            }}

            // Fullscreen toggle handler
            document.getElementById('btnFullscreen').addEventListener('click', function() {{
                const container = document.getElementById('container');
                if (!document.fullscreenElement) {{
                    container.requestFullscreen().catch(err => {{
                        console.error('Fullscreen request error:', err);
                    }});
                }} else {{
                    document.exitFullscreen();
                }}
            }});

            function initWebGL() {{
                const container = document.getElementById('container');
                const canvas = document.getElementById('canvas3d');
                const width = container.clientWidth;
                const height = container.clientHeight;

                // 1. Three.js Scene
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xF8FAFC);
                scene.fog = new THREE.Fog(0xF8FAFC, 12, 35);

                // 2. Perspective Camera
                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
                const initialCamPos = new THREE.Vector3(2.2, 1.4, 2.8);
                camera.position.copy(initialCamPos);

                // 3. WebGL Renderer
                const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                renderer.outputEncoding = THREE.sRGBEncoding;

                // 4. OrbitControls
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.target.set(0, 0, 0);
                controls.maxPolarAngle = Math.PI / 2 + 0.1;
                controls.minDistance = 0.5;
                controls.maxDistance = 15;

                // 5. Studio Lighting
                const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.85);
                scene.add(ambientLight);

                const dirLight1 = new THREE.DirectionalLight(0xFFFFFF, 1.0);
                dirLight1.position.set(5, 10, 7);
                dirLight1.castShadow = true;
                dirLight1.shadow.mapSize.width = 1024;
                dirLight1.shadow.mapSize.height = 1024;
                scene.add(dirLight1);

                const dirLight2 = new THREE.DirectionalLight(0x93C5FD, 0.5);
                dirLight2.position.set(-5, 4, -5);
                scene.add(dirLight2);

                const dirLight3 = new THREE.DirectionalLight(0xFFFFFF, 0.4);
                dirLight3.position.set(0, -5, 2);
                scene.add(dirLight3);

                // 6. Ground Grid
                const gridHelper = new THREE.GridHelper(10, 20, 0xCBD5E1, 0xE2E8F0);
                gridHelper.position.y = -1.05;
                scene.add(gridHelper);

                // State
                let currentModel = null;
                let autoRotate = false;
                let isWireframe = false;
                const palette = ['{default_color}', '#1E293B', '#64748B', '#D97706', '#10B981', '#7C3AED'];
                let paletteIndex = 0;

                const modelFormat = "{model_format}";
                const rawModelData = {model_data_js};

                if (modelFormat === "glb" && rawModelData) {{
                    loadGLB(rawModelData);
                }} else if (modelFormat === "obj" && rawModelData) {{
                    loadOBJ(rawModelData);
                }}

                // GLB Loader using GLTFLoader
                function loadGLB(base64Str) {{
                    try {{
                        const binaryStr = atob(base64Str);
                        const len = binaryStr.length;
                        const bytes = new Uint8Array(len);
                        for (let i = 0; i < len; i++) {{
                            bytes[i] = binaryStr.charCodeAt(i);
                        }}

                        const loader = new THREE.GLTFLoader();
                        loader.parse(bytes.buffer, '', function(gltf) {{
                            currentModel = gltf.scene;

                            let vTotal = 0;
                            let fTotal = 0;

                            currentModel.traverse(function(child) {{
                                if (child.isMesh) {{
                                    child.castShadow = true;
                                    child.receiveShadow = true;
                                    if (child.geometry) {{
                                        const pos = child.geometry.attributes.position;
                                        if (pos) vTotal += pos.count;
                                        if (child.geometry.index) {{
                                            fTotal += child.geometry.index.count / 3;
                                        }} else if (pos) {{
                                            fTotal += pos.count / 3;
                                        }}
                                    }}
                                }}
                            }});

                            // Normalize bounding box & center
                            const box = new THREE.Box3().setFromObject(currentModel);
                            const center = box.getCenter(new THREE.Vector3());
                            const size = box.getSize(new THREE.Vector3());
                            const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
                            const scale = 2.0 / maxDim;

                            currentModel.scale.set(scale, scale, scale);
                            currentModel.position.sub(center.multiplyScalar(scale));

                            scene.add(currentModel);
                            document.getElementById('statsHud').innerText = `Vertices: ${{vTotal}} | Faces: ${{Math.round(fTotal)}}`;
                        }}, function(err) {{
                            console.error('GLTF parse error:', err);
                        }});
                    }} catch (e) {{
                        console.error('GLB decode error:', e);
                    }}
                }}

                // OBJ Loader using in-memory parser
                function loadOBJ(text) {{
                    const lines = text.split('\\n');
                    const rawVerts = [];
                    const indices = [];

                    for (let i = 0; i < lines.length; i++) {{
                        const line = lines[i].trim();
                        if (!line || line.startsWith('#')) continue;
                        const parts = line.split(/\\s+/);
                        if (parts[0] === 'v') {{
                            rawVerts.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
                        }} else if (parts[0] === 'f') {{
                            const faceVerts = [];
                            for (let j = 1; j < parts.length; j++) {{
                                const vIndex = parseInt(parts[j].split('/')[0]) - 1;
                                faceVerts.push(vIndex);
                            }}
                            if (faceVerts.length >= 3) {{
                                indices.push(faceVerts[0], faceVerts[1], faceVerts[2]);
                                if (faceVerts.length === 4) {{
                                    indices.push(faceVerts[0], faceVerts[2], faceVerts[3]);
                                }}
                            }}
                        }}
                    }}

                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(rawVerts, 3));
                    if (indices.length > 0) {{
                        geometry.setIndex(indices);
                    }}
                    geometry.computeVertexNormals();
                    geometry.center();

                    // Automatic viewport fitting and bounding box normalization
                    geometry.computeBoundingBox();
                    if (geometry.boundingBox) {{
                        const size = new THREE.Vector3();
                        geometry.boundingBox.getSize(size);
                        const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
                        const scale = 2.0 / maxDim;
                        geometry.scale(scale, scale, scale);
                        geometry.center();
                    }}

                    const mat = new THREE.MeshStandardMaterial({{
                        color: new THREE.Color(palette[paletteIndex]),
                        roughness: 0.35,
                        metalness: 0.20,
                        wireframe: isWireframe,
                        side: THREE.DoubleSide
                    }});

                    currentModel = new THREE.Mesh(geometry, mat);
                    currentModel.castShadow = true;
                    currentModel.receiveShadow = true;
                    scene.add(currentModel);

                    const vCount = geometry.attributes.position ? geometry.attributes.position.count : 0;
                    const fCount = geometry.index ? geometry.index.count / 3 : vCount / 3;
                    document.getElementById('statsHud').innerText = `Vertices: ${{vCount}} | Faces: ${{Math.round(fCount)}}`;
                }}

                // Toolbar handlers
                const btnRot = document.getElementById('btnRotate');
                if (btnRot) {{
                    btnRot.addEventListener('click', function() {{
                        autoRotate = !autoRotate;
                        this.classList.toggle('active', autoRotate);
                    }});
                }}

                const btnWire = document.getElementById('btnWireframe');
                if (btnWire) {{
                    btnWire.addEventListener('click', function() {{
                        isWireframe = !isWireframe;
                        this.classList.toggle('active', isWireframe);
                        if (currentModel) {{
                            currentModel.traverse(function(child) {{
                                if (child.isMesh && child.material) {{
                                    if (Array.isArray(child.material)) {{
                                        child.material.forEach(m => m.wireframe = isWireframe);
                                    }} else {{
                                        child.material.wireframe = isWireframe;
                                    }}
                                }}
                            }});
                        }}
                    }});
                }}

                const btnCol = document.getElementById('btnColor');
                if (btnCol) {{
                    btnCol.addEventListener('click', function() {{
                        paletteIndex = (paletteIndex + 1) % palette.length;
                        const newCol = new THREE.Color(palette[paletteIndex]);
                        if (currentModel) {{
                            currentModel.traverse(function(child) {{
                                if (child.isMesh && child.material) {{
                                    if (Array.isArray(child.material)) {{
                                        child.material.forEach(m => m.color.copy(newCol));
                                    }} else {{
                                        child.material.color.copy(newCol);
                                    }}
                                }}
                            }});
                        }}
                    }});
                }}

                const btnRes = document.getElementById('btnReset');
                if (btnRes) {{
                    btnRes.addEventListener('click', function() {{
                        camera.position.copy(initialCamPos);
                        controls.target.set(0, 0, 0);
                        controls.update();
                    }});
                }}

                window.addEventListener('resize', function() {{
                    const w = container.clientWidth;
                    const h = container.clientHeight;
                    camera.aspect = w / h;
                    camera.updateProjectionMatrix();
                    renderer.setSize(w, h);
                }}, false);

                function animate() {{
                    requestAnimationFrame(animate);
                    if (autoRotate && currentModel) {{
                        currentModel.rotation.y += 0.008;
                    }}
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)
