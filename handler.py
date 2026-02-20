import fal
from fal.container import ContainerImage
from fal.toolkit import Image, download_model_weights
from fastapi import Response, HTTPException
from pathlib import Path
import json
import uuid
import base64
import requests
import websocket
import traceback
import os
import copy
import random
import tempfile
import time
import subprocess
import sys
import logging
import warnings
from io import BytesIO
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from comfy_models import MODEL_LIST
from workflow import WORKFLOW_JSON

# Suppress urllib and fal toolkit warnings
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("fal.toolkit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="fal.toolkit")

# -------------------------------------------------
# Container setup
# -------------------------------------------------
PWD = Path(__file__).resolve().parent if __file__ else Path(os.getcwd())
dockerfile_path = f"{PWD}/Dockerfile"
custom_image = ContainerImage.from_dockerfile(dockerfile_path)

COMFY_HOST = "127.0.0.1:8188"
DEBUG_LOGS = os.environ.get("FAL_DEBUG") == "1"

def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(message)

# -------------------------------------------------
# Presets
# -------------------------------------------------
PRESETS = {
    "imperfect_skin": {"cfg": 0.1, "denoise": 0.34, "resolution": 2048},
    "high_end_skin": {"cfg": 1.1, "denoise": 0.30, "resolution": 3072},
    "smooth_skin": {
        "cfg": 1.1,
        "denoise": 0.30,
        "resolution": 2048,
        "prompt_override": True,
        "positive_prompt": (
            "ultra realistic portrait of [subject], flawless clear face, "
            "smooth radiant skin texture, fine pores, balanced complexion, "
            "healthy glow, cinematic lighting"
        ),
        "negative_prompt": (
            "freckles, spots, blemishes, acne, pigmentation, redness, "
            "rough skin, waxy skin, plastic texture, airbrushed"
        )
    },
    "portrait": {"cfg": 0.5, "denoise": 0.35, "resolution": 2048},
    "mid_range": {"cfg": 1.4, "denoise": 0.40, "resolution": 2048},
    "full_body": {"cfg": 1.5, "denoise": 0.30, "resolution": 2048},
}

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def check_server(url, retries=500, delay=0.1):
    for _ in range(retries):
        try:
            if requests.get(url).status_code == 200:
                return True
        except:
            pass
        time.sleep(delay)
    return False

def fal_image_to_base64(img: Image) -> str:
    pil = img.to_pil()
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def image_url_to_base64(image_url: str) -> str:
    """Download image from URL and convert to base64."""
    response = requests.get(image_url)
    response.raise_for_status()
    pil = PILImage.open(BytesIO(response.content))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def upload_images(images):
    for img in images:
        blob = base64.b64decode(img["image"])
        files = {"image": (img["name"], BytesIO(blob), "image/png")}
        r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        r.raise_for_status()

# -------------------------------------------------
# Input Model
# -------------------------------------------------
class SkinFixInput(BaseModel):
    image_url: str = Field(
        ...,
        title="Input Image",
        description="URL of the image to enhance and upscale."
    )

    mode: Literal["preset", "custom"] = Field(
        default="preset",
        title="Configuration Mode",
        description="Choose 'preset' to use predefined settings or 'custom' for manual control"
    )

    preset_name: Optional[Literal[
        "imperfect_skin",
        "high_end_skin",
        "smooth_skin",
        "portrait",
        "mid_range",
        "full_body"
    ]] = Field(
        default="high_end_skin",
        title="Preset",
        description="Select a preset (only active when mode is 'preset')"
    )

    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        title="Skin Realism",
        description="Adjust skin realism (only active when mode is 'custom')"
    )

    skin_refinement: int = Field(
        default=30,
        ge=0,
        le=100,
        title="Skin Refinement",
        description="Adjust skin refinement level (only active when mode is 'custom')"
    )

    seed: int = Field(default=123456789, title="Random Seed")

    upscale_resolution: Literal[
        1024, 1280, 1536, 1792,
        2048, 2304, 2560, 2816, 3072
    ] = Field(
        default=2048,
        title="Upscaler Resolution",
        description="Target resolution for upscaling (only active when mode is 'custom')"
    )

# -------------------------------------------------
# Output Model
# -------------------------------------------------
class SkinFixOutput(BaseModel):
    images: list[Image] = Field(
        description="Output images from skin fix processing"
    )

# -------------------------------------------------
# App
# -------------------------------------------------
class SkinFixApp(
    fal.App,
    keep_alive=100,
    min_concurrency=0,
    max_concurrency=5,
    name="skin-fix-app2",
):
    """Skin Fix - Advanced skin refinement and upscaling."""
    
    image = custom_image
    machine_type = "GPU-H100"
    requirements = ["websockets", "websocket-client"]

    # 🔒 CRITICAL
    private_logs = True  # Set to True if logs may contain sensitive info (e.g. image URLs)

    def setup(self):
        # Print GPU info
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True
            ).strip()
            debug_log(f"🖥️ GPU Type: {gpu_info}")
        except Exception as e:
            debug_log(f"⚠️ Could not detect GPU: {e}")

        # Download models using fal.toolkit (Caching!)
        for model in MODEL_LIST:
            try:
                debug_log(f"⬇️ Downloading: {model['url']}")
                # This uses fal's internal cache
                cached_path = download_model_weights(model["url"])
                
                # Symlink to ComfyUI models dir
                target_path = model["target"]
                ensure_dir(target_path)
                
                if os.path.exists(target_path) or os.path.islink(target_path):
                    os.unlink(target_path)  # Remove existing to be safe
                    
                os.symlink(cached_path, target_path)
                debug_log(f"✅ Linked: {cached_path} -> {target_path}")
                
            except Exception as e:
                debug_log(f"❌ Failed to download/link {model['url']}: {e}")
                raise e

        # Preflight: verify face_parsing node is present and importable
        try:
            node_dir = "/comfyui/custom_nodes/comfyui_face_parsing"
            debug_log(f"🧩 face_parsing node dir exists: {os.path.isdir(node_dir)}")
            if "/comfyui" not in sys.path:
                sys.path.insert(0, "/comfyui")
            import importlib
            fp_module = importlib.import_module("custom_nodes.comfyui_face_parsing")
            node_map = getattr(fp_module, "NODE_CLASS_MAPPINGS", {})
            has_parser = "FaceParsingResultsParser(FaceParsing)" in node_map
            debug_log(f"🧩 face_parsing node registered: {has_parser}")
        except Exception as e:
            debug_log(f"❌ face_parsing import failed: {e}")

        # Start ComfyUI (NO --log-stdout)
        self.comfy = subprocess.Popen(
            [
                "python", "-u", "/comfyui/main.py",
                "--disable-auto-launch",
                "--disable-metadata",
                "--listen", "--port", "8188"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if not check_server(f"http://{COMFY_HOST}/system_stats"):
            raise RuntimeError("ComfyUI failed to start")

        # Verify ComfyUI registered the face_parsing node
        try:
            info = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
            info.raise_for_status()
            nodes = info.json()
            if "FaceParsingResultsParser(FaceParsing)" not in nodes:
                raise RuntimeError("FaceParsingResultsParser(FaceParsing) not in object_info")
            debug_log("✅ ComfyUI reports FaceParsingResultsParser(FaceParsing) is available")
        except Exception as e:
            raise RuntimeError(f"ComfyUI missing face_parsing node: {e}")

        # -------------------------------------------------
        # Warmup
        # -------------------------------------------------
        self._run_warmup()

    def _run_warmup(self):
        """Run a lightweight generation to load models into GPU memory."""
        debug_log("🔥 Starting warmup...")
        try:
            # 1. Create a dummy image (black 64x64)
            dummy_img = PILImage.new("RGB", (64, 64), (0, 0, 0))
            buf = BytesIO()
            dummy_img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            
            # 2. Upload dummy image
            image_name = f"warmup_{uuid.uuid4().hex}.png"
            upload_images([{
                "name": image_name,
                "image": image_b64
            }])

            # 3. Prepare warmup workflow
            # We clone the workflow and set minimal parameters for speed
            # but enough to trigger model loading.
            job = copy.deepcopy(WORKFLOW_JSON)
            workflow = job["input"]["workflow"]
            
            # Set input image
            workflow["545"]["inputs"]["image"] = image_name
            
            # Set minimal processing parameters
            workflow["510"]["inputs"]["steps"] = 1  # 1 step is enough to load models
            workflow["548"]["inputs"]["resolution"] = 512
            workflow["548"]["inputs"]["max_resolution"] = 512
            workflow["549"]["inputs"]["encode_tile_size"] = 512
            workflow["549"]["inputs"]["decode_tile_size"] = 512
            
            # 4. Run ComfyUI
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

            resp = requests.post(
                f"http://{COMFY_HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30
            )
            
            if resp.status_code == 200:
                prompt_id = resp.json()["prompt_id"]
                # Wait for completion
                while True:
                    out = ws.recv()
                    if not out.strip().startswith('{'):
                        continue
                    msg = json.loads(out)
                    if msg.get("type") == "executing" and msg["data"]["node"] is None:
                        break
                debug_log("✅ Warmup complete - models loaded.")
            else:
                debug_log(f"⚠️ Warmup failed to queue: {resp.text}")
                
            ws.close()
            
        except Exception as e:
            debug_log(f"⚠️ Warmup failed: {e}")
            traceback.print_exc()

    @fal.endpoint("/")
    async def handler(self, input: SkinFixInput, response: Response) -> SkinFixOutput:
        try:
            job = copy.deepcopy(WORKFLOW_JSON)
            workflow = job["input"]["workflow"]

            # -------------------------------------------------
            # 1️⃣ Download and read input image resolution (KEY PART)
            # -------------------------------------------------
            # Download image from URL
            image_b64 = image_url_to_base64(input.image_url)
            
            # Get resolution from downloaded image
            pil_img = PILImage.open(BytesIO(base64.b64decode(image_b64)))
            w, h = pil_img.size
            input_image_resolution = max(w, h)

            # Upload image
            image_name = f"input_{uuid.uuid4().hex}.png"
            upload_images([{
                "name": image_name,
                "image": image_b64
            }])
            workflow["545"]["inputs"]["image"] = image_name

            sampler = workflow["510"]["inputs"]

            # -------------------------------------------------
            # 2️⃣ Apply settings based on mode
            # -------------------------------------------------
            if input.mode == "preset":
                # Use preset settings
                p = PRESETS[input.preset_name]
                target_resolution = max(p["resolution"], input_image_resolution)
                sampler["cfg"] = p["cfg"]
                sampler["denoise"] = p["denoise"]

                if p.get("prompt_override"):
                    workflow["506"]["inputs"]["part1"] = p["positive_prompt"]
                    workflow["507"]["inputs"]["text"] = p["negative_prompt"]
            else:
                # Use custom settings
                sampler["cfg"] = input.cfg
                sampler["denoise"] = 0.30 + (input.skin_refinement / 100.0) * 0.10
                target_resolution = max(input.upscale_resolution, input_image_resolution)

            # Apply seed
            sampler["seed"] = input.seed

            # -------------------------------------------------
            # 3️⃣ Apply resolution to SeedVR2 nodes
            # -------------------------------------------------
            workflow["548"]["inputs"]["resolution"] = target_resolution
            workflow["548"]["inputs"]["max_resolution"] = 4096
            workflow["549"]["inputs"]["encode_tile_size"] = min(1024, target_resolution)
            workflow["549"]["inputs"]["decode_tile_size"] = min(1024, target_resolution)

            # Always randomize SeedVR2 internal seed
            workflow["548"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

            # -------------------------------------------------
            # 4️⃣ Run ComfyUI
            # -------------------------------------------------
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

            resp = requests.post(
                f"http://{COMFY_HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30
            )
            
            # Log detailed error if request fails
            if resp.status_code != 200:
                error_detail = resp.text
                debug_log(f"ComfyUI Error Response: {error_detail}")
                raise HTTPException(status_code=500, detail=f"ComfyUI rejected workflow: {error_detail}")
            
            prompt_id = resp.json()["prompt_id"]

            while True:
                out = ws.recv()
                if not out.strip().startswith('{'):
                    continue  # Skip non-JSON messages (progress messages)
                msg = json.loads(out)
                if msg.get("type") == "executing" and msg["data"]["node"] is None:
                    break

            history = requests.get(
                f"http://{COMFY_HOST}/history/{prompt_id}"
            ).json()

            images = []
            for node in history[prompt_id]["outputs"].values():
                for img in node.get("images", []):
                    params = (
                        f"filename={img['filename']}"
                        f"&subfolder={img.get('subfolder','')}"
                        f"&type={img['type']}"
                    )
                    r = requests.get(f"http://{COMFY_HOST}/view?{params}")
                    pil_image = PILImage.open(BytesIO(r.content))
                    output_image = Image.from_pil(pil_image, format="png")
                    images.append(output_image)

            ws.close()
            
            # Set billing headers
            response.headers["x-fal-billable-units"] = str(len(images))
            
            return SkinFixOutput(images=images)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
