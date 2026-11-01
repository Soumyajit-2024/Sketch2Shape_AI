import os
import sys

# Entrypoint wrapper to support running from workspace root:
# python -m streamlit run app/app.py

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(ROOT_DIR, "Sketch2Shape_AI")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Clear cached project modules so that modifications take effect immediately
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("modules.") or mod_name == "modules":
        del sys.modules[mod_name]

app_target = os.path.join(PROJECT_DIR, "app", "app.py")
if os.path.exists(app_target):
    with open(app_target, "r", encoding="utf-8") as f:
        code = compile(f.read(), app_target, "exec")
        exec(code, globals())
else:
    raise FileNotFoundError(f"Could not find application file at {app_target}")
