"""
MotionCanvas AI - Turn Your Imagination Into Motion.

Image-to-video studio for Streamlit. The heavy model (WarmBloodAban/Minimax-h3_Singularity)
runs on YOUR external ComfyUI GPU backend; this app only uploads the image, injects your
prompt into your ComfyUI workflow, submits it, monitors the real job and fetches the real MP4.

Streamlit secrets (.streamlit/secrets.toml locally, or App settings > Secrets on Streamlit Cloud):

    GROQ_API_KEY    = "gsk_..."                    # prompt enhancement (Groq)
    HF_TOKEN        = "hf_..."                     # Hugging Face repo check
    COMFYUI_API_URL = "https://your-backend-url"   # must be reachable from the internet
    # Optional:
    COMFYUI_API_KEY = "..."                        # sent as a Bearer token to your backend/proxy
    APP_PASSWORD    = "..."                        # password gate so strangers cannot use your GPU
"""

from __future__ import annotations

import copy
import hmac
import html
import io
import json
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
APP_NAME = "MotionCanvas AI"
TAGLINE = "Turn Your Imagination Into Motion."
MODEL_REPO = "WarmBloodAban/Minimax-h3_Singularity"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 10
MIN_SIDE, MAX_SIDE = 64, 8192
HISTORY_LIMIT = 8
POLL_INTERVAL = 2.0
WORKFLOW_DIR = Path(__file__).parent / "workflows"

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

PARAM_SPECS: dict[str, dict[str, Any]] = {
    "image": {"label": "Input image", "type": "str", "required": True},
    "prompt": {"label": "Positive prompt", "type": "str", "required": True},
    "negative_prompt": {"label": "Negative prompt", "type": "str", "required": False},
    "seed": {"label": "Seed", "type": "int", "required": False},
    "frames": {"label": "Frame count", "type": "int", "required": False},
    "fps": {"label": "Frames per second", "type": "int", "required": False},
    "steps": {"label": "Sampling steps", "type": "int", "required": False},
    "cfg": {"label": "CFG scale", "type": "float", "required": False},
    "motion_strength": {"label": "Motion strength", "type": "float", "required": False},
}
NUMERIC_HINTS = {
    "seed": ("seed", "noise_seed"),
    "frames": ("length", "num_frames", "frames", "video_frames"),
    "fps": ("fps", "frame_rate"),
    "steps": ("steps",),
    "cfg": ("cfg",),
    "motion_strength": ("motion_bucket_id", "motion_strength", "motion_scale", "motion"),
}
TEXT_KEYS = ("text", "prompt", "positive_prompt", "text_g")

VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".gif": "image/gif",
}
VIDEO_RANK = {".mp4": 0, ".webm": 1, ".mov": 2, ".mkv": 3, ".gif": 4}

AUTO = "Auto (let the model decide)"
SUBJECT_MOVES = [AUTO, "Subtle breathing and natural blinking", "Gentle head turn toward the camera",
                 "Walking slowly forward", "Hair and clothing flowing in the wind",
                 "Confident expressive gesture", "Dynamic action movement"]
ENV_MOVES = [AUTO, "Drifting clouds and wind moving through foliage", "Rippling water and gentle waves",
             "Falling rain with shimmering reflections", "Slowly drifting snow",
             "Floating dust particles in light shafts", "Swirling fog and mist",
             "Busy background activity"]
CAMERAS = [AUTO, "Static locked-off camera", "Slow push-in", "Slow pull-out", "Smooth pan left",
           "Smooth pan right", "Tilt up", "Tilt down", "Orbit around the subject",
           "Subtle handheld motion", "Cinematic dolly zoom"]
LIGHTING = [AUTO, "Golden hour warm light", "Soft diffused daylight", "Dramatic low-key lighting",
            "Neon glow at night", "Moody backlight with rim light", "Studio softbox lighting",
            "Moonlit blue tones"]
ATMOSPHERES = [AUTO, "Dreamy and serene", "Epic and dramatic", "Mysterious and moody",
               "Warm and nostalgic", "Energetic and vibrant", "Calm and meditative",
               "Tense and suspenseful"]
STYLES = [AUTO, "Photorealistic cinematic", "Film noir", "Anime-inspired",
          "35mm film with fine grain", "Hyper-real commercial", "Documentary realism",
          "Fantasy illustration coming alive"]
INTENSITIES = ["Subtle", "Moderate", "Dynamic", "Intense"]

SUGGESTIONS = [
    ("Portrait comes alive", "The person slowly turns toward the camera with a soft smile, hair drifting in a light breeze, eyes catching the warm light as the camera gently pushes in."),
    ("Epic landscape", "Clouds roll across the sky as golden light sweeps over the landscape, grass and trees swaying in the wind while the camera glides slowly forward."),
    ("Rainy neon street", "Rain falls steadily through glowing neon reflections, passersby move softly in the background, and the camera drifts forward at street level."),
    ("Ocean at sunset", "Waves roll gently toward the shore, foam shimmering in the last golden light, sea birds gliding past as the camera slowly pulls back."),
    ("Product hero shot", "The product rotates slowly on a polished surface as soft highlights glide across its edges, with a smooth cinematic orbit around it."),
    ("Fantasy scene", "Glowing particles float through the air, mist curls along the ground, and the character's cloak ripples as the camera performs a slow dramatic push-in."),
]

ENHANCER_SYSTEM = (
    "You are an expert prompt engineer for image-to-video diffusion models. Rewrite the user's "
    "description into ONE cinematic prompt of 60 to 110 words describing how the provided still image "
    "should come to life: subject motion, environment motion, camera movement, lighting, atmosphere and "
    "visual style. Use concrete verbs and keep motion physically plausible. Do not invent new subjects or "
    "change the scene; the image defines the appearance. Output ONLY the prompt text, with no preface, "
    "quotes or markdown."
)

PERSIST_KEYS = {
    "prompt_idea", "final_prompt", "negative_prompt", "opt_subject", "opt_env", "opt_camera",
    "opt_light", "opt_atmos", "opt_style", "opt_intensity",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
  color: #ede9fe;
  background:
    radial-gradient(1100px 600px at 8% -10%, rgba(124,58,237,.30), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(168,85,247,.18), transparent 55%),
    linear-gradient(180deg, #07040f 0%, #0b0617 55%, #05030a 100%);
}
.stApp p, .stApp label, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] { color: #ede9fe; }
.stApp small, .stApp [data-testid="stCaptionContainer"] { color: #a79bd0 !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stSidebar"] {
  background: rgba(14, 8, 28, .88); backdrop-filter: blur(18px);
  border-right: 1px solid rgba(167,139,250,.16);
}
.mc-hero h1 {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -.02em; margin: .2rem 0 .3rem 0;
  background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 45%, #a855f7 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
}
.mc-hero p { color: #b8aee0; font-size: 1.05rem; margin: 0 0 1.2rem 0; }
.mc-eyebrow {
  display: inline-block; padding: .25rem .7rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: #ddd6fe;
  background: rgba(124,58,237,.22); border: 1px solid rgba(167,139,250,.35);
}
.mc-card {
  background: rgba(255,255,255,.045); border: 1px solid rgba(167,139,250,.18);
  border-radius: 20px; padding: 1.15rem 1.3rem; margin-bottom: .9rem;
  backdrop-filter: blur(14px); box-shadow: 0 10px 40px rgba(0,0,0,.35);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.mc-card:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.42); box-shadow: 0 14px 44px rgba(124,58,237,.18); }
.mc-card h4 { margin: 0 0 .35rem 0; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #a79bd0 !important; font-weight: 600; }
.mc-card .mc-value { font-size: 1.35rem; font-weight: 700; color: #ffffff; margin-bottom: .2rem; }
.mc-card .mc-sub { color: #a79bd0; font-size: .85rem; }
.mc-section { font-size: 1.05rem; font-weight: 700; margin: .4rem 0 .6rem 0; color: #f5f3ff; }
.mc-step { display: flex; gap: .8rem; align-items: flex-start; margin-bottom: .6rem; }
.mc-step .n { min-width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; background: linear-gradient(135deg,#7c3aed,#a855f7); color: #fff; }
.mc-badge { display: inline-block; padding: .18rem .6rem; border-radius: 999px; font-size: .75rem; font-weight: 600; margin-right: .3rem; }
.mc-ok { background: rgba(34,197,94,.15); color: #86efac; border: 1px solid rgba(34,197,94,.35); }
.mc-warn { background: rgba(250,204,21,.12); color: #fde68a; border: 1px solid rgba(250,204,21,.35); }
.mc-bad { background: rgba(244,63,94,.14); color: #fda4af; border: 1px solid rgba(244,63,94,.35); }
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid rgba(167,139,250,.3); background: rgba(255,255,255,.05);
  color: #ede9fe; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #a78bfa; transform: translateY(-1px); color: #fff; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #7c3aed, #a855f7); border: none; color: #fff; font-weight: 600;
  box-shadow: 0 8px 24px rgba(124,58,237,.4);
}
.stButton > button:disabled { opacity: .45; }
.stTextArea textarea, .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.05) !important; color: #f5f3ff !important;
  border-radius: 12px !important; border: 1px solid rgba(167,139,250,.25) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(124,58,237,.08); border: 1.5px dashed rgba(167,139,250,.5); border-radius: 16px;
}
[data-testid="stExpander"] { border: 1px solid rgba(167,139,250,.2); border-radius: 14px; background: rgba(255,255,255,.03); }
video, img { border-radius: 16px; }
@media (max-width: 768px) {
  .mc-hero h1 { font-size: 1.8rem; }
  .block-container { padding: 1rem .8rem; }
}
"""


class AppError(Exception):
    """An error with a message that is safe and useful to show to the user."""


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def secret(name: str) -> str:
    """Read a secret: session override > Streamlit secrets > environment variable."""
    if name != "APP_PASSWORD":
        override = st.session_state.get("overrides", {}).get(name, "")
        if override:
            return override.strip()
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets file present
        value = ""
    return str(value or os.environ.get(name, "")).strip()


def secret_source(name: str) -> str:
    if st.session_state.get("overrides", {}).get(name):
        return "session"
    try:
        if st.secrets.get(name, ""):
            return "secrets"
    except Exception:
        pass
    return "env" if os.environ.get(name) else "missing"


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "workflow"


def badge(text: str, kind: str = "ok") -> str:
    return f'<span class="mc-badge mc-{kind}">{html.escape(text)}</span>'


def card(title: str, value: str, sub: str = "") -> str:
    return (f'<div class="mc-card"><h4>{html.escape(title)}</h4><div class="mc-value">{value}</div>'
            f'<div class="mc-sub">{sub}</div></div>')


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="mc-hero"><span class="mc-eyebrow">{APP_NAME}</span>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def queue_set(**values: Any) -> None:
    st.session_state.setdefault("_pending", {}).update(values)


def apply_pending() -> None:
    pending = st.session_state.pop("_pending", None)
    if pending:
        for key, value in pending.items():
            st.session_state[key] = value


def set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


def keep_widget_state() -> None:
    """Streamlit drops widget state when a widget is not rendered; this keeps it across pages."""
    for key in list(st.session_state.keys()):
        if key in PERSIST_KEYS or key.startswith("param_"):
            st.session_state[key] = st.session_state[key]


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    defaults: dict[str, Any] = {
        "history": [], "workflows": {}, "active_workflow": "", "image": None,
        "uploader_nonce": 0, "replacing": False, "overrides": {}, "backend_status": None,
        "groq_model": DEFAULT_GROQ_MODEL, "timeout_min": 20, "authed": False,
        "prompt_idea": "", "final_prompt": "", "negative_prompt": "",
        "opt_subject": AUTO, "opt_env": AUTO, "opt_camera": AUTO, "opt_light": AUTO,
        "opt_atmos": AUTO, "opt_style": AUTO, "opt_intensity": "Moderate",
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)
    if not ss.get("_preloaded"):
        ss["_preloaded"] = True
        preload_workflows()


def preload_workflows() -> None:
    """Optionally load workflow JSON files committed to a ./workflows folder (no secrets inside!)."""
    if not WORKFLOW_DIR.is_dir():
        return
    for path in sorted(WORKFLOW_DIR.glob("*.json"))[:10]:
        try:
            name, graph, mapping = parse_workflow_upload(path.read_bytes(), path.name)
            add_workflow(name, graph, mapping)
        except (AppError, OSError):
            continue


# --------------------------------------------------------------------------------------
# Workflow handling
# --------------------------------------------------------------------------------------
def is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def literal_inputs(node: dict) -> dict:
    return {k: v for k, v in (node.get("inputs") or {}).items() if not is_link(v)}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sorted_ids(graph: dict) -> list[str]:
    return sorted(graph.keys(), key=lambda k: (0, int(k), "") if k.isdigit() else (1, 0, k))


def node_label(graph: dict, nid: str) -> str:
    node = graph[nid]
    cls = node.get("class_type", "?")
    title = (node.get("_meta") or {}).get("title", "")
    return f"{nid} · {cls}" + (f" — {title}" if title and title != cls else "")


def type_ok(value: Any, ptype: str) -> bool:
    if ptype == "str":
        return isinstance(value, str)
    if ptype == "int":
        return is_number(value) and float(value).is_integer()
    return is_number(value)


def mappable_keys(node: dict, ptype: str) -> list[str]:
    return [k for k, v in literal_inputs(node).items() if type_ok(v, ptype)]


def binding_ok(graph: dict, binding: Optional[dict], param: str) -> bool:
    if not binding or binding.get("node") not in graph:
        return False
    node = graph[binding["node"]]
    key = binding.get("key")
    return key in literal_inputs(node) and type_ok(node["inputs"][key], PARAM_SPECS[param]["type"])


def get_input(graph: dict, binding: dict) -> Any:
    return graph[binding["node"]]["inputs"][binding["key"]]


def workflow_issues(entry: Optional[dict]) -> list[str]:
    if not entry:
        return ["No workflow selected. Upload your ComfyUI API-format workflow in Model Settings."]
    issues = []
    for param, spec in PARAM_SPECS.items():
        if spec["required"] and not binding_ok(entry["graph"], entry["mapping"].get(param), param):
            issues.append(f"Workflow mapping for '{spec['label']}' is missing or invalid (Model Settings).")
    return issues


def validate_api_workflow(graph: Any) -> None:
    if isinstance(graph, dict) and "nodes" in graph and "links" in graph:
        raise AppError(
            "This looks like a UI-format workflow. In ComfyUI enable Dev Mode and use "
            "'Save (API Format)' (or 'Export (API)') to get the JSON this app needs."
        )
    if not isinstance(graph, dict) or not graph:
        raise AppError("The workflow JSON must be an object of node IDs (API format).")
    for nid, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            raise AppError(f"Node '{nid}' is not valid API format (needs 'class_type' and 'inputs').")


def parse_workflow_upload(raw: bytes, filename: str) -> tuple[str, dict, Optional[dict]]:
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppError(f"'{filename}' is not valid JSON: {exc}") from exc
    mapping = None
    name = Path(filename).stem
    if isinstance(data, dict) and data.get("motioncanvas_bundle"):
        graph, mapping, name = data.get("workflow"), data.get("mapping") or {}, data.get("name") or name
    else:
        graph = data
    validate_api_workflow(graph)
    return str(name), graph, mapping if isinstance(mapping, dict) else None


def add_workflow(name: str, graph: dict, mapping: Optional[dict] = None) -> str:
    workflows = st.session_state.workflows
    base, i = name, 2
    while name in workflows:
        name = f"{base} ({i})"
        i += 1
    workflows[name] = {"graph": graph, "mapping": mapping if mapping is not None else autodetect_mapping(graph)}
    st.session_state.active_workflow = name
    return name


def active_entry() -> Optional[dict]:
    return st.session_state.workflows.get(st.session_state.active_workflow)


def find_text_node(graph: dict, nid: Any, role: str, depth: int = 8, seen: Optional[set] = None):
    seen = seen if seen is not None else set()
    nid = str(nid)
    if nid in seen or depth < 0 or nid not in graph:
        return None
    seen.add(nid)
    node = graph[nid]
    if "textencode" in node.get("class_type", "").lower():
        lit = literal_inputs(node)
        for key in TEXT_KEYS:
            if isinstance(lit.get(key), str):
                return nid, key
    inputs = node.get("inputs") or {}
    keys = [role] if is_link(inputs.get(role)) else [k for k, v in inputs.items() if is_link(v)]
    for key in keys:
        found = find_text_node(graph, inputs[key][0], role, depth - 1, seen)
        if found:
            return found
    return None


def autodetect_mapping(graph: dict) -> dict:
    """Heuristic suggestions only - the user should verify them in Model Settings."""
    mapping: dict[str, dict] = {}
    ids = sorted_ids(graph)
    for nid in ids:
        node = graph[nid]
        if "loadimage" in node.get("class_type", "").lower() and isinstance(literal_inputs(node).get("image"), str):
            mapping["image"] = {"node": nid, "key": "image"}
            break
    for nid in ids:
        inputs = graph[nid].get("inputs") or {}
        if is_link(inputs.get("positive")) and is_link(inputs.get("negative")):
            pos = find_text_node(graph, inputs["positive"][0], "positive")
            neg = find_text_node(graph, inputs["negative"][0], "negative")
            if pos:
                mapping["prompt"] = {"node": pos[0], "key": pos[1]}
                if neg and neg != pos:
                    mapping["negative_prompt"] = {"node": neg[0], "key": neg[1]}
                break
    if "prompt" not in mapping:
        encoders = [n for n in ids if "textencode" in graph[n].get("class_type", "").lower()]
        for param, nid in zip(("prompt", "negative_prompt"), encoders[:2]):
            for key in TEXT_KEYS:
                if isinstance(literal_inputs(graph[nid]).get(key), str):
                    mapping[param] = {"node": nid, "key": key}
                    break
    for param, names in NUMERIC_HINTS.items():
        done = False
        for nid in ids:
            lit = literal_inputs(graph[nid])
            for key in names:
                if key in lit and type_ok(lit[key], PARAM_SPECS[param]["type"]):
                    mapping[param] = {"node": nid, "key": key}
                    done = True
                    break
            if done:
                break
    return mapping


def build_workflow(graph: dict, mapping: dict, values: dict) -> dict:
    wf = copy.deepcopy(graph)
    for param, value in values.items():
        if value is None or not binding_ok(graph, mapping.get(param), param):
            continue
        ptype = PARAM_SPECS[param]["type"]
        value = int(value) if ptype == "int" else float(value) if ptype == "float" else str(value)
        b = mapping[param]
        wf[b["node"]]["inputs"][b["key"]] = value
    return wf


# --------------------------------------------------------------------------------------
# ComfyUI backend client
# --------------------------------------------------------------------------------------
def describe_comfy_error(data: Any, fallback: str) -> str:
    if not isinstance(data, dict):
        return fallback
    parts: list[str] = []
    err = data.get("error")
    if isinstance(err, dict):
        parts += [str(err.get("message") or err.get("type") or ""), str(err.get("details") or "")]
    elif isinstance(err, str):
        parts.append(err)
    for nid, info in (data.get("node_errors") or {}).items():
        for e in (info or {}).get("errors", []):
            parts.append(f"Node {nid} ({info.get('class_type', 'node')}): {e.get('message', '')} {e.get('details', '')}".strip())
    return "\n".join(p for p in parts if p) or fallback


class ComfyClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base = normalize_url(base_url)
        self.http = requests.Session()
        self.http.headers["ngrok-skip-browser-warning"] = "true"
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, timeout=(10, 60), **kwargs) -> requests.Response:
        try:
            return self.http.request(method, f"{self.base}{path}", timeout=timeout, **kwargs)
        except requests.exceptions.SSLError as exc:
            raise AppError("SSL error talking to the ComfyUI backend. Check that COMFYUI_API_URL uses the right https:// address.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise AppError("Could not reach the ComfyUI backend. Is the GPU server running and is COMFYUI_API_URL publicly reachable? "
                           "(Streamlit Cloud cannot reach localhost or private network addresses.)") from exc
        except requests.exceptions.Timeout as exc:
            raise AppError("The ComfyUI backend timed out.") from exc
        except requests.RequestException as exc:
            raise AppError(f"Backend request failed: {exc}") from exc

    def _json(self, resp: requests.Response, what: str) -> Any:
        if resp.status_code in (401, 403):
            raise AppError(f"The backend rejected the request for {what} (HTTP {resp.status_code}). Check COMFYUI_API_KEY / proxy authentication.")
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code != 200:
            raise AppError(f"Backend error during {what} (HTTP {resp.status_code}):\n" +
                           describe_comfy_error(data, resp.text[:300] or "no details returned"))
        if data is None:
            raise AppError(f"The backend returned a non-JSON response for {what}. The URL may point to a proxy or login page instead of ComfyUI.")
        return data

    def system_stats(self) -> dict:
        return self._json(self._request("GET", "/system_stats"), "the connection check")

    def object_info(self) -> dict:
        return self._json(self._request("GET", "/object_info", timeout=(10, 180)), "the node check")

    def upload_image(self, name: str, data: bytes, mime: str) -> str:
        resp = self._request("POST", "/upload/image", timeout=(10, 180),
                             files={"image": (name, data, mime)}, data={"overwrite": "true", "type": "input"})
        info = self._json(resp, "the image upload")
        sub = info.get("subfolder", "")
        return f"{sub}/{info['name']}" if sub else info["name"]

    def queue_prompt(self, workflow: dict) -> str:
        resp = self._request("POST", "/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex})
        data = self._json(resp, "submitting the workflow")
        if "prompt_id" not in data:
            raise AppError("The backend accepted the request but returned no job id.")
        return data["prompt_id"]

    def history(self, prompt_id: str) -> dict:
        return self._json(self._request("GET", f"/history/{prompt_id}"), "reading job status")

    def queue(self) -> dict:
        return self._json(self._request("GET", "/queue"), "reading the queue")

    def view(self, filename: str, subfolder: str, ftype: str) -> bytes:
        resp = self._request("GET", "/view", timeout=(10, 600),
                             params={"filename": filename, "subfolder": subfolder, "type": ftype})
        if resp.status_code != 200:
            raise AppError(f"Could not download '{filename}' from the backend (HTTP {resp.status_code}).")
        return resp.content

    def interrupt(self) -> None:
        resp = self._request("POST", "/interrupt")
        if resp.status_code != 200:
            raise AppError(f"Interrupt failed (HTTP {resp.status_code}).")


def get_client() -> ComfyClient:
    url = secret("COMFYUI_API_URL")
    if not url:
        raise AppError("COMFYUI_API_URL is not configured. Add it to your Streamlit secrets (see API Configuration).")
    return ComfyClient(url, secret("COMFYUI_API_KEY"))


def find_video_files(outputs: Any) -> list[dict]:
    found = []
    for out in (outputs or {}).values():
        if not isinstance(out, dict):
            continue
        for items in out.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    if os.path.splitext(item["filename"])[1].lower() in VIDEO_RANK:
                        found.append(item)
    found.sort(key=lambda i: (i.get("type", "output") != "output", VIDEO_RANK[os.path.splitext(i["filename"])[1].lower()]))
    return found


def execution_error_text(status: dict) -> str:
    for message in status.get("messages", []):
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error":
            d = message[1] or {}
            return (f"The backend failed while running node {d.get('node_id', '?')} ({d.get('node_type', '?')}): "
                    f"{d.get('exception_message', 'unknown error')}".strip())
    return "The backend reported an error while running this workflow. Check the ComfyUI server logs."


def queue_state(client: ComfyClient, prompt_id: str) -> str:
    q = client.queue()
    if any(len(i) > 1 and i[1] == prompt_id for i in q.get("queue_running", [])):
        return "running on the GPU"
    for pos, item in enumerate(q.get("queue_pending", []), start=1):
        if len(item) > 1 and item[1] == prompt_id:
            return f"queued (position {pos})"
    return "waiting for the backend"


def wait_for_job(client: ComfyClient, prompt_id: str, timeout_s: int, line) -> dict:
    start, failures = time.time(), 0
    while True:
        elapsed = int(time.time() - start)
        if elapsed > timeout_s:
            raise AppError(f"Timed out after {timeout_s // 60} minutes. The job may still be running on your backend; "
                           "check Generation History later to fetch the result.")
        state = "checking status"
        try:
            entry = client.history(prompt_id).get(prompt_id)
            failures = 0
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise AppError(execution_error_text(status))
                if entry.get("outputs") or status.get("completed"):
                    return entry
                state = "running on the GPU"
            else:
                state = queue_state(client, prompt_id)
        except AppError as exc:
            if "failed while running node" in str(exc) or "error while running" in str(exc):
                raise
            failures += 1
            if failures >= 5:
                raise
            state = "retrying after a connection hiccup"
        line.markdown(f"⏱ **{elapsed // 60:02d}:{elapsed % 60:02d}** · {state}")
        time.sleep(POLL_INTERVAL)


def fetch_video(client: ComfyClient, history_entry: dict) -> dict:
    files = find_video_files(history_entry.get("outputs"))
    if not files:
        nodes = ", ".join(sorted((history_entry.get("outputs") or {}).keys())) or "none"
        raise AppError("The job finished but produced no video file. Make sure your workflow ends in a video-saving node "
                       f"(e.g. SaveVideo or VHS Video Combine). Output nodes returned: {nodes}.")
    item = files[0]
    data = client.view(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
    ext = os.path.splitext(item["filename"])[1].lower()
    return {"bytes": data, "name": item["filename"], "mime": VIDEO_MIME[ext], "ext": ext}


# --------------------------------------------------------------------------------------
# Groq + Hugging Face
# --------------------------------------------------------------------------------------
def groq_enhance(idea: str, direction: str) -> str:
    key = secret("GROQ_API_KEY")
    if not key:
        raise AppError("GROQ_API_KEY is not configured. Add it to your Streamlit secrets (see API Configuration).")
    payload = {
        "model": st.session_state.groq_model or DEFAULT_GROQ_MODEL,
        "temperature": 0.7,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": ENHANCER_SYSTEM},
            {"role": "user", "content": f"Description: {idea}\nPreferred direction: {direction or 'none - decide sensibly'}"},
        ],
    }
    try:
        resp = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=45)
    except requests.exceptions.Timeout as exc:
        raise AppError("Groq timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise AppError(f"Could not reach Groq: {exc}") from exc
    if resp.status_code == 401:
        raise AppError("Groq rejected the API key (401). Check GROQ_API_KEY.")
    if resp.status_code == 429:
        raise AppError("Groq rate limit reached (429). Wait a moment and try again.")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:200])
        except ValueError:
            detail = resp.text[:200]
        raise AppError(f"Groq error (HTTP {resp.status_code}): {detail}")
    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise AppError("Groq returned an unexpected response.") from exc
    text = text.strip("`").strip().strip('"').strip()
    if not text:
        raise AppError("Groq returned an empty prompt. Try again.")
    return text


def check_hf_repo() -> dict:
    token = secret("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{MODEL_REPO}", headers=headers, timeout=20)
    except requests.RequestException as exc:
        raise AppError(f"Could not reach Hugging Face: {exc}") from exc
    if resp.status_code in (401, 403):
        raise AppError("Hugging Face denied access. The repo may be gated or private; check that HF_TOKEN is valid and has access.")
    if resp.status_code == 404:
        raise AppError(f"The repository '{MODEL_REPO}' was not found (or needs a token with access).")
    if resp.status_code != 200:
        raise AppError(f"Hugging Face returned HTTP {resp.status_code}.")
    return resp.json()


def direction_text() -> str:
    ss = st.session_state
    parts = []
    for label, key in (("subject movement", "opt_subject"), ("environment movement", "opt_env"),
                       ("camera", "opt_camera"), ("lighting", "opt_light"),
                       ("atmosphere", "opt_atmos"), ("visual style", "opt_style")):
        if ss[key] != AUTO:
            parts.append(f"{label}: {ss[key]}")
    parts.append(f"motion intensity: {ss['opt_intensity'].lower()}")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------
# Image validation
# --------------------------------------------------------------------------------------
def validate_image(uploaded) -> dict:
    data = uploaded.getvalue()
    if not data:
        raise AppError("The file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AppError(f"The image is {fmt_bytes(len(data))}; the limit is {MAX_UPLOAD_MB} MB.")
    ext = os.path.splitext(uploaded.name)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        raise AppError("Unsupported file type. Use JPG, JPEG, PNG or WEBP.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            fmt, (w, h) = img.format, img.size
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise AppError("This file is not a valid, readable image.") from exc
    if fmt not in ("JPEG", "PNG", "WEBP"):
        raise AppError(f"The file content is {fmt}, not JPG/PNG/WEBP.")
    if min(w, h) < MIN_SIDE:
        raise AppError(f"The image is too small ({w}×{h}). Minimum side is {MIN_SIDE}px.")
    if max(w, h) > MAX_SIDE:
        raise AppError(f"The image is too large ({w}×{h}). Maximum side is {MAX_SIDE}px.")
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[fmt]
    return {"name": uploaded.name, "bytes": data, "width": w, "height": h, "size": len(data), "format": fmt,
            "mime": mime, "upload_name": f"motioncanvas_{uuid.uuid4().hex[:10]}.{ext}"}


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
def render_video(record: dict, key: str) -> None:
    video = record.get("video")
    if not video:
        return
    if video["mime"].startswith("video/") and video["mime"] != "video/x-matroska":
        st.video(video["bytes"], format=video["mime"])
    elif video["mime"] == "image/gif":
        st.image(video["bytes"])
    else:
        st.info("This format cannot be previewed in the browser. Use the download button.")
    st.download_button(f"Download {video['ext'].lstrip('.').upper()}", data=video["bytes"],
                       file_name=f"motioncanvas_{record['stamp']}{video['ext']}", mime=video["mime"],
                       key=f"dl_{key}", icon=":material/download:")


def add_history(record: dict) -> None:
    hist = st.session_state.history
    hist.append(record)
    while len(hist) > HISTORY_LIMIT:
        hist.pop(0)


def check_backend() -> None:
    try:
        stats = get_client().system_stats()
        dev = (stats.get("devices") or [{}])[0]
        info = {"ok": True, "gpu": dev.get("name", "unknown device"), "vram": dev.get("vram_total"),
                "version": (stats.get("system") or {}).get("comfyui_version", "")}
    except AppError as exc:
        info = {"ok": False, "msg": str(exc)}
    info["time"] = datetime.now().strftime("%H:%M:%S")
    st.session_state.backend_status = info


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
PAGES: dict[str, Any] = {}


def page_home() -> None:
    page_header(APP_NAME, TAGLINE)
    ss = st.session_state
    entry = active_entry()
    bs = ss.backend_status
    if not secret("COMFYUI_API_URL"):
        backend = (badge("Not configured", "bad"), "Add COMFYUI_API_URL in secrets")
    elif bs is None:
        backend = (badge("Not checked", "warn"), "Use the button below to test")
    elif bs["ok"]:
        backend = (badge("Online", "ok"), html.escape(f"{bs['gpu']} · checked {bs['time']}"))
    else:
        backend = (badge("Unreachable", "bad"), f"checked {bs['time']}")
    groq = (badge("Ready", "ok"), "Prompt enhancement available") if secret("GROQ_API_KEY") else \
           (badge("Missing key", "warn"), "Prompt enhancement disabled")
    if entry and not workflow_issues(entry):
        wf = (badge("Ready", "ok"), html.escape(ss.active_workflow))
    elif entry:
        wf = (badge("Needs mapping", "warn"), html.escape(ss.active_workflow))
    else:
        wf = (badge("None loaded", "bad"), "Upload one in Model Settings")
    done = sum(1 for r in ss.history if r.get("video"))
    cols = st.columns(4)
    for col, (title, (val, sub)) in zip(cols, (("GPU backend", backend), ("Groq", groq), ("Workflow", wf),
                                               ("Videos this session", (f"{done}", "History is kept for this browser session")))):
        col.markdown(card(title, val, sub), unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Start creating", type="primary", icon=":material/movie_creation:"):
        st.switch_page(PAGES["create"])
    if c2.button("Check backend", icon=":material/dns:"):
        check_backend()
        st.rerun()
    if bs and not bs["ok"]:
        st.error(bs["msg"])

    st.markdown('<div class="mc-section">How it works</div>', unsafe_allow_html=True)
    steps = [("Upload", "Drop in a JPG, PNG or WEBP. It is validated before anything is sent."),
             ("Direct the motion", "Describe the movement, enhance it with Groq, then edit the final prompt."),
             ("Generate", "Your image and prompt are injected into your ComfyUI workflow and run on your GPU."),
             ("Download", "The real MP4 produced by the backend is shown and ready to download.")]
    left, right = st.columns(2)
    for i, (title, text) in enumerate(steps, start=1):
        (left if i % 2 else right).markdown(
            f'<div class="mc-card"><div class="mc-step"><div class="n">{i}</div>'
            f"<div><b>{title}</b><br><span class='mc-sub'>{text}</span></div></div></div>", unsafe_allow_html=True)


def page_create() -> None:
    apply_pending()
    page_header("Create Video", "Upload an image, direct the motion, and generate a real video on your GPU backend.")
    ss = st.session_state
    left, right = st.columns([1, 1.25], gap="large")

    # ---- 1. Image Studio -------------------------------------------------------------
    with left:
        st.markdown('<div class="mc-section">1 · Image Studio</div>', unsafe_allow_html=True)
        img = ss.image
        if img is None or ss.replacing:
            uploaded = st.file_uploader("Drag and drop an image here", type=ALLOWED_EXT,
                                        key=f"uploader_{ss.uploader_nonce}",
                                        help=f"JPG, JPEG, PNG or WEBP · up to {MAX_UPLOAD_MB} MB")
            if uploaded is not None:
                try:
                    ss.image = validate_image(uploaded)
                    ss.replacing = False
                    ss.uploader_nonce += 1
                    st.rerun()
                except AppError as exc:
                    st.error(str(exc))
            if img is not None and st.button("Cancel replace"):
                ss.replacing = False
                st.rerun()
        else:
            st.image(img["bytes"], caption=img["name"])
            st.markdown(badge(f"{img['width']}×{img['height']} px", "ok") + badge(fmt_bytes(img["size"]), "ok") +
                        badge(img["format"], "ok"), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button("Replace image", icon=":material/swap_horiz:"):
                ss.replacing = True
                st.rerun()
            if b2.button("Remove image", icon=":material/delete:"):
                ss.image = None
                ss.uploader_nonce += 1
                st.rerun()

    # ---- 2. Prompt Studio ------------------------------------------------------------
    with right:
        st.markdown('<div class="mc-section">2 · AI Prompt Studio</div>', unsafe_allow_html=True)
        st.text_area("Describe how the picture should move", key="prompt_idea", height=100,
                     placeholder="e.g. She turns toward the camera while the wind moves her hair")
        with st.expander("Cinematic prompt suggestions"):
            cols = st.columns(3)
            for i, (title, text) in enumerate(SUGGESTIONS):
                cols[i % 3].button(title, key=f"sugg_{i}", on_click=set_state, args=("prompt_idea", text))

        st.markdown("**Prompt settings**")
        s1, s2, s3 = st.columns(3)
        s1.selectbox("Subject movement", SUBJECT_MOVES, key="opt_subject")
        s2.selectbox("Environment movement", ENV_MOVES, key="opt_env")
        s3.selectbox("Camera movement", CAMERAS, key="opt_camera")
        s4, s5, s6 = st.columns(3)
        s4.selectbox("Lighting", LIGHTING, key="opt_light")
        s5.selectbox("Atmosphere", ATMOSPHERES, key="opt_atmos")
        s6.selectbox("Visual style", STYLES, key="opt_style")
        st.select_slider("Motion intensity", INTENSITIES, key="opt_intensity",
                         help="Described to the model through the prompt.")

        a1, a2, a3 = st.columns(3)
        if a1.button("Enhance with Groq", type="primary", icon=":material/auto_awesome:",
                     disabled=not ss.prompt_idea.strip()):
            try:
                with st.spinner("Groq is writing your cinematic prompt…"):
                    queue_set(final_prompt=groq_enhance(ss.prompt_idea.strip(), direction_text()))
                st.rerun()
            except AppError as exc:
                st.error(str(exc))
        if a2.button("Add settings to prompt", icon=":material/playlist_add:",
                     disabled=not (ss.final_prompt.strip() or ss.prompt_idea.strip())):
            base = ss.final_prompt.strip() or ss.prompt_idea.strip()
            queue_set(final_prompt=f"{base.rstrip('.')}. Direction: {direction_text()}.")
            st.rerun()
        a3.button("Clear prompt", icon=":material/backspace:",
                  on_click=lambda: ss.update(prompt_idea="", final_prompt="", negative_prompt=""))
        st.text_area("Final prompt (editable: this exact text is sent to the model)", key="final_prompt", height=150)

    # ---- 3. Video settings -----------------------------------------------------------
    st.markdown('<div class="mc-section">3 · Video Settings</div>', unsafe_allow_html=True)
    names = list(ss.workflows.keys())
    values: dict[str, Any] = {}
    entry = None
    if not names:
        st.warning("No workflow loaded yet. Upload your ComfyUI API-format workflow in Model Settings.")
        if st.button("Open Model Settings", icon=":material/tune:"):
            st.switch_page(PAGES["model"])
    else:
        if ss.active_workflow not in names:
            ss.active_workflow = names[0]
        ss.active_workflow = st.selectbox("Workflow", names, index=names.index(ss.active_workflow))
        entry = active_entry()
        graph, mapping = entry["graph"], entry["mapping"]
        slug = slugify(ss.active_workflow)
        supported = [PARAM_SPECS[p]["label"] for p in PARAM_SPECS if binding_ok(graph, mapping.get(p), p)]
        st.caption("Settings exposed by this workflow: " + (", ".join(supported) or "none"))
        if binding_ok(graph, mapping.get("negative_prompt"), "negative_prompt"):
            cur = get_input(graph, mapping["negative_prompt"])
            if "negative_prompt_loaded" not in ss.get("_flags", {}).get(slug, {}):
                ss.setdefault("_flags", {}).setdefault(slug, {})["negative_prompt_loaded"] = True
                if not ss.negative_prompt:
                    ss.negative_prompt = cur
            st.text_area("Negative prompt", key="negative_prompt", height=80)
        numeric = [p for p in PARAM_SPECS if p not in ("image", "prompt", "negative_prompt")
                   and binding_ok(graph, mapping.get(p), p)]
        cols = st.columns(3) if numeric else []
        for i, param in enumerate(numeric):
            spec, cur, key = PARAM_SPECS[param], get_input(graph, mapping[param]), f"param_{slug}_{param}"
            with cols[i % 3]:
                if spec["type"] == "int":
                    ss.setdefault(key, max(0, min(int(cur), 2**53 - 1)))
                    if param == "seed":
                        rand_key = f"{key}_random"
                        ss.setdefault(rand_key, True)
                        st.checkbox("Randomize seed each run", key=rand_key)
                        st.number_input(spec["label"], key=key, min_value=0, step=1, disabled=ss[rand_key])
                        values[param] = random.randint(0, 2**32 - 1) if ss[rand_key] else ss[key]
                    else:
                        st.number_input(spec["label"], key=key, min_value=0, step=1)
                        values[param] = ss[key]
                else:
                    ss.setdefault(key, float(cur))
                    st.number_input(spec["label"], key=key, step=0.1, format="%.2f")
                    values[param] = ss[key]
        if entry and workflow_issues(entry):
            for issue in workflow_issues(entry):
                st.warning(issue)

    # ---- 4. Generate ------------------------------------------------------------------
    st.markdown('<div class="mc-section">4 · Generate</div>', unsafe_allow_html=True)
    prompt = (ss.final_prompt.strip() or ss.prompt_idea.strip())
    issues = []
    if ss.image is None:
        issues.append("Upload an image.")
    if not prompt:
        issues.append("Write a prompt.")
    if not secret("COMFYUI_API_URL"):
        issues.append("Set COMFYUI_API_URL in your secrets.")
    issues += workflow_issues(entry)
    for issue in issues:
        st.markdown(badge(issue, "warn"), unsafe_allow_html=True)
    if not ss.final_prompt.strip() and ss.prompt_idea.strip():
        st.caption("No final prompt yet: your description will be sent as written.")
    if st.button("Generate video", type="primary", disabled=bool(issues), icon=":material/rocket_launch:"):
        negative = ss.negative_prompt if entry and binding_ok(entry["graph"], entry["mapping"].get("negative_prompt"), "negative_prompt") else None
        run_generation(entry, prompt, negative, values)


def run_generation(entry: dict, prompt: str, negative: Optional[str], values: dict) -> None:
    ss = st.session_state
    image = ss.image
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {"id": "", "stamp": stamp, "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "prompt": prompt,
              "workflow": ss.active_workflow, "status": "preparing", "message": "", "video": None, "seconds": None}
    started = time.time()
    with st.status("Generating on your GPU backend…", expanded=True) as status:
        try:
            client = get_client()
            st.write("Uploading image to ComfyUI…")
            ref = client.upload_image(image["upload_name"], image["bytes"], image["mime"])
            wf = build_workflow(entry["graph"], entry["mapping"],
                                {**values, "image": ref, "prompt": prompt, "negative_prompt": negative})
            st.write("Submitting workflow…")
            record["id"] = client.queue_prompt(wf)
            record["status"] = "submitted"
            add_history(record)
            st.write(f"Job `{record['id']}` accepted. Keep this tab open while it runs.")
            line = st.empty()
            hist_entry = wait_for_job(client, record["id"], int(ss.timeout_min) * 60, line)
            st.write("Downloading the finished video…")
            record["video"] = fetch_video(client, hist_entry)
            record["status"], record["seconds"] = "completed", int(time.time() - started)
            status.update(label=f"Video ready in {record['seconds']}s", state="complete", expanded=False)
        except AppError as exc:
            record["status"], record["message"] = "failed", str(exc)
            if record not in ss.history and record["id"]:
                add_history(record)
            status.update(label="Generation failed", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # unexpected: do not hide it
            record["status"], record["message"] = "failed", f"Unexpected error: {exc}"
            status.update(label="Unexpected error", state="error")
            st.error(f"Unexpected error: {exc}")
            return
    st.success("Your video is ready.")
    render_video(record, f"new_{record['stamp']}")


def page_history() -> None:
    page_header("Generation History", "Videos from this browser session. Download what you want to keep.")
    ss = st.session_state
    if not ss.history:
        st.info("Nothing here yet. Generate a video from Create Video.")
        return
    kinds = {"completed": "ok", "submitted": "warn", "failed": "bad"}
    for i, rec in enumerate(reversed(ss.history)):
        uid = f"{rec['stamp']}_{i}"
        with st.expander(f"{rec['created']} · {rec['prompt'][:70]}", expanded=(i == 0)):
            st.markdown(badge(rec["status"].title(), kinds.get(rec["status"], "warn")) +
                        badge(rec["workflow"], "ok") + (badge(f"{rec['seconds']}s", "ok") if rec["seconds"] else ""),
                        unsafe_allow_html=True)
            st.write(rec["prompt"])
            if rec["message"]:
                st.error(rec["message"])
            render_video(rec, uid)
            c1, c2 = st.columns(2)
            if not rec["video"] and rec["id"] and c1.button("Check backend for result", key=f"chk_{uid}"):
                try:
                    client = get_client()
                    hist = client.history(rec["id"]).get(rec["id"])
                    if not hist:
                        st.info("The backend has no finished result for this job yet (it may be queued or have been cleared).")
                    elif (hist.get("status") or {}).get("status_str") == "error":
                        rec["status"], rec["message"] = "failed", execution_error_text(hist["status"])
                        st.rerun()
                    else:
                        rec["video"], rec["status"], rec["message"] = fetch_video(client, hist), "completed", ""
                        st.rerun()
                except AppError as exc:
                    st.error(str(exc))
            if c2.button("Remove from history", key=f"rm_{uid}"):
                ss.history.remove(rec)
                st.rerun()
    if st.button("Clear all history"):
        ss.history = []
        st.rerun()


def page_model() -> None:
    page_header("Model Settings", "Model, GPU backend and the ComfyUI workflow that drives generation.")
    ss = st.session_state

    st.markdown('<div class="mc-section">Model</div>', unsafe_allow_html=True)
    st.markdown(card("Hugging Face repository", f'<a href="https://huggingface.co/{MODEL_REPO}" target="_blank" style="color:#c4b5fd">{MODEL_REPO}</a>',
                     "Runs on your ComfyUI GPU backend, never inside Streamlit. The model files and any custom nodes it needs must be installed on that backend."),
                unsafe_allow_html=True)
    if st.button("Verify repository on Hugging Face", icon=":material/verified:"):
        try:
            info = check_hf_repo()
            files = [s.get("rfilename", "") for s in info.get("siblings", [])]
            st.success(f"Repository found: {info.get('id', MODEL_REPO)} · gated: {info.get('gated', False)} · "
                       f"private: {info.get('private', False)} · {len(files)} files")
            if files:
                st.code("\n".join(files[:25]) + ("\n…" if len(files) > 25 else ""))
        except AppError as exc:
            st.error(str(exc))

    st.markdown('<div class="mc-section">GPU backend (ComfyUI)</div>', unsafe_allow_html=True)
    st.markdown(badge("URL configured", "ok") if secret("COMFYUI_API_URL") else badge("COMFYUI_API_URL missing", "bad"),
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("Test connection", icon=":material/dns:"):
        check_backend()
        bs = ss.backend_status
        st.success(f"Connected · {bs['gpu']}" + (f" · ComfyUI {bs['version']}" if bs["version"] else "")) if bs["ok"] else st.error(bs["msg"])
    if b2.button("Interrupt the running job", icon=":material/stop_circle:"):
        try:
            get_client().interrupt()
            st.success("Interrupt sent to the backend.")
        except AppError as exc:
            st.error(str(exc))

    st.markdown('<div class="mc-section">Workflow configuration</div>', unsafe_allow_html=True)
    st.caption("Upload a workflow exported from ComfyUI in API format (Dev Mode > Save (API Format)), or a MotionCanvas bundle saved below.")
    up = st.file_uploader("Workflow JSON", type=["json"], key=f"wf_up_{ss.uploader_nonce}")
    if up is not None:
        try:
            name, graph, mapping = parse_workflow_upload(up.getvalue(), up.name)
            add_workflow(name, graph, mapping)
            ss.uploader_nonce += 1
            st.rerun()
        except AppError as exc:
            st.error(str(exc))
    names = list(ss.workflows.keys())
    if not names:
        st.info("No workflow loaded yet.")
        return
    if ss.active_workflow not in names:
        ss.active_workflow = names[0]
    ss.active_workflow = st.selectbox("Active workflow", names, index=names.index(ss.active_workflow), key="model_wf_select")
    name, entry = ss.active_workflow, active_entry()
    graph, mapping, slug = entry["graph"], entry["mapping"], slugify(ss.active_workflow)

    st.markdown("**Node mapping.** Choose which node input receives each value. Only mapped settings appear in Create Video.")
    labels = {nid: node_label(graph, nid) for nid in graph}
    none_opt = "— not mapped —"
    for param, spec in PARAM_SPECS.items():
        current = mapping.get(param) or {}
        c1, c2 = st.columns(2)
        options = [none_opt] + sorted_ids(graph)
        idx = options.index(current["node"]) if current.get("node") in options else 0
        node = c1.selectbox(f"{spec['label']}{' *' if spec['required'] else ''} · node", options, index=idx,
                            key=f"mapnode_{slug}_{param}", format_func=lambda x: x if x == none_opt else labels[x])
        if node == none_opt:
            mapping.pop(param, None)
            c2.caption("Not mapped")
            continue
        keys = mappable_keys(graph[node], spec["type"])
        if not keys:
            mapping.pop(param, None)
            c2.warning(f"Node {node} has no editable {spec['type']} inputs.")
            continue
        kidx = keys.index(current["key"]) if current.get("node") == node and current.get("key") in keys else 0
        key = c2.selectbox("Input key", keys, index=kidx, key=f"mapkey_{slug}_{param}_{node}")
        mapping[param] = {"node": node, "key": key}

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Auto-detect", icon=":material/search:", help="Heuristic suggestions. Always verify them."):
        entry["mapping"] = autodetect_mapping(graph)
        for k in [k for k in ss.keys() if k.startswith((f"mapnode_{slug}_", f"mapkey_{slug}_", f"param_{slug}_"))]:
            del ss[k]
        st.rerun()
    bundle = json.dumps({"motioncanvas_bundle": 1, "name": name, "workflow": graph, "mapping": mapping}, indent=2)
    a2.download_button("Save bundle", bundle, file_name=f"{slug}.motioncanvas.json", mime="application/json",
                       icon=":material/save:", help="Workflow + mapping in one file. Re-upload it next time.")
    if a3.button("Validate on backend", icon=":material/fact_check:"):
        try:
            missing = sorted({n["class_type"] for n in graph.values()} - set(get_client().object_info().keys()))
            st.success("All workflow nodes exist on the backend.") if not missing else \
                st.error("These node types are missing on your ComfyUI backend (install the required custom nodes): " + ", ".join(missing))
        except AppError as exc:
            st.error(str(exc))
    if a4.button("Delete workflow", icon=":material/delete:"):
        del ss.workflows[name]
        ss.active_workflow = ""
        st.rerun()
    for issue in workflow_issues(entry):
        st.warning(issue)
    ss.timeout_min = st.slider("Generation timeout (minutes)", 2, 60, int(ss.timeout_min))


def page_api() -> None:
    page_header("API Configuration", "Keys live in Streamlit secrets and are never displayed here.")
    ss = st.session_state
    rows = [("GROQ_API_KEY", "Prompt enhancement (Groq)", True), ("HF_TOKEN", "Hugging Face repository check", False),
            ("COMFYUI_API_URL", "Your ComfyUI GPU backend", True), ("COMFYUI_API_KEY", "Optional backend/proxy token", False),
            ("APP_PASSWORD", "Optional password gate for this app", False)]
    labels = {"secrets": ("Configured via secrets", "ok"), "env": ("Configured via environment", "ok"),
              "session": ("Session override active", "warn")}
    for name, desc, required in rows:
        src = secret_source(name) if name != "APP_PASSWORD" else ("secrets" if secret(name) else "missing")
        text, kind = labels.get(src, ("Missing" if required else "Not set (optional)", "bad" if required else "warn"))
        st.markdown(f"**`{name}`** · {desc} &nbsp; " + badge(text, kind), unsafe_allow_html=True)

    st.markdown('<div class="mc-section">Groq model</div>', unsafe_allow_html=True)
    ss.groq_model = st.text_input("Model used for prompt enhancement", value=ss.groq_model,
                                  help="Any chat model available on your Groq account.").strip() or DEFAULT_GROQ_MODEL

    with st.expander("Session-only overrides (not saved anywhere)"):
        st.caption("Useful for testing. These live only in this browser session. Prefer Streamlit secrets for real deployments.")
        for name in ("GROQ_API_KEY", "HF_TOKEN", "COMFYUI_API_URL", "COMFYUI_API_KEY"):
            val = st.text_input(name, type="password", key=f"w_ovr_{name}")
            if val.strip():
                ss.overrides[name] = val.strip()
        if st.button("Clear overrides"):
            ss.overrides = {}
            for name in ("GROQ_API_KEY", "HF_TOKEN", "COMFYUI_API_URL", "COMFYUI_API_KEY"):
                ss.pop(f"w_ovr_{name}", None)
            st.rerun()

    st.markdown('<div class="mc-section">Secrets template</div>', unsafe_allow_html=True)
    st.code('GROQ_API_KEY = "gsk_your_key_here"\nHF_TOKEN = "hf_your_token_here"\n'
            'COMFYUI_API_URL = "https://your-gpu-backend.example.com"\n'
            '# optional\nCOMFYUI_API_KEY = ""\nAPP_PASSWORD = ""', language="toml")
    st.caption("Streamlit Cloud: App settings → Secrets. Local: .streamlit/secrets.toml (keep it out of GitHub).")


def page_about() -> None:
    page_header("About MotionCanvas", TAGLINE)
    st.markdown(card("What it is", "A thin, honest studio in front of your GPU",
                     "MotionCanvas AI never simulates results. Every video you see was produced by the model on your own ComfyUI backend."),
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(card("Pipeline", "Image → Prompt → ComfyUI → MP4",
                     "Upload to /upload/image, workflow to /prompt, status via /history and /queue, video via /view."), unsafe_allow_html=True)
    c2.markdown(card("Model", html.escape(MODEL_REPO),
                     "Hosted by you on a ComfyUI-compatible GPU. Hugging Face does not provide hosted inference for it."), unsafe_allow_html=True)
    st.markdown("""
**Good to know**
- The backend must be reachable from the internet (Streamlit Cloud cannot reach `localhost`). RunPod, a cloud VM, or a secured tunnel all work.
- Workflows must be exported in ComfyUI **API format**. Node IDs and input keys are configurable in Model Settings.
- History lives in your browser session. Download videos you want to keep.
- Protect your GPU: set `APP_PASSWORD` in secrets if the app is public.
""")


# --------------------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------------------
def require_login() -> None:
    expected = secret("APP_PASSWORD")
    if not expected or st.session_state.authed:
        return
    page_header(APP_NAME, "Enter the password to continue.")
    with st.form("login"):
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Unlock", type="primary"):
            if hmac.compare_digest(pwd.encode(), expected.encode()):
                st.session_state.authed = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


def main() -> None:
    init_state()
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    require_login()
    keep_widget_state()
    PAGES.update(
        home=st.Page(page_home, title="Home Dashboard", icon=":material/home:", default=True),
        create=st.Page(page_create, title="Create Video", icon=":material/movie_creation:"),
        history=st.Page(page_history, title="Generation History", icon=":material/history:"),
        model=st.Page(page_model, title="Model Settings", icon=":material/tune:"),
        api=st.Page(page_api, title="API Configuration", icon=":material/key:"),
        about=st.Page(page_about, title="About MotionCanvas", icon=":material/info:"),
    )
    nav = st.navigation(list(PAGES.values()))
    with st.sidebar:
        st.markdown(f"### 🎬 {APP_NAME}")
        st.caption(TAGLINE)
    nav.run()


if __name__ == "__main__":
    main()
