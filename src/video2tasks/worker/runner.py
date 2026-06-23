"""Worker runner implementation."""

import time
import json
import base64
from io import BytesIO
from typing import Optional, Dict, Any

import requests
import numpy as np
from PIL import Image

from ..config import Config
from ..vlm import create_backend
from ..prompt import prompt_switch_detection

MAX_LOCAL_RETRIES = 2


def _is_empty_vlm_json(vlm_json: Optional[Dict[str, Any]]) -> bool:
    return (not isinstance(vlm_json, dict)) or (not vlm_json)


def decode_b64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string to numpy BGR array."""
    if not b64_str:
        return None
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Convert RGB to BGR for OpenCV compatibility
        rgb_array = np.array(img)
        bgr_array = rgb_array[:, :, ::-1]
        return bgr_array
    except Exception:
        return None


def run_worker(config: Config) -> None:
    """Run the worker loop."""
    server_url = config.worker.server_url
    
    # Create and warmup backend
    backend_kwargs = {}
    if config.worker.backend == "qwen3vl":
        backend_kwargs = {
            "model_path": config.worker.qwen3vl.model_path,
            "device_map": config.worker.qwen3vl.device_map,
        }
    elif config.worker.backend == "remote_api":
        backend_kwargs = {
            "url": config.worker.remote_api.api_url,
            "api_key": config.worker.remote_api.api_key,
            "headers": config.worker.remote_api.headers,
            "timeout_sec": config.worker.remote_api.timeout_sec,
        }
    elif config.worker.backend == "openai_chat":
        backend_kwargs = {
            "base_url": config.worker.openai_chat.base_url,
            "api_key": config.worker.openai_chat.api_key,
            "model": config.worker.openai_chat.model,
            "timeout_sec": config.worker.openai_chat.timeout_sec,
            "max_tokens": config.worker.openai_chat.max_tokens,
            "headers": config.worker.openai_chat.headers,
        }
    
    backend = create_backend(config.worker.backend, **backend_kwargs)
    print(f"[Worker] Using backend: {backend.name}")
    backend.warmup()
    
    print(f"[Worker] Connecting to {server_url}")
    
    max_connection_retries = 30  # ~60 seconds total
    connection_retry_count = 0
    
    try:
        while True:
            try:
                # Get job
                try:
                    r = requests.get(f"{server_url}/get_job", timeout=60)
                    connection_retry_count = 0  # Reset on successful connection
                except requests.exceptions.RequestException as e:
                    connection_retry_count += 1
                    if connection_retry_count >= max_connection_retries:
                        print(f"[Worker] Failed to connect after {max_connection_retries} retries. Exiting.")
                        break
                    print(f"[Worker] Waiting for server at {server_url}... (attempt {connection_retry_count}/{max_connection_retries})")
                    time.sleep(2)
                    continue
                
                if r.status_code != 200:
                    time.sleep(0.5)
                    continue
                
                resp = r.json()
                if resp.get("status") == "empty":
                    time.sleep(0.5)
                    continue
                
                job = resp.get("data")
                if job is None:
                    print("[Worker] Invalid job data received")
                    time.sleep(1)
                    continue
                task_id = job.get("task_id", "unknown")
                
                # Decode images
                images_b64 = job.get("images", [])
                images = []
                for b64 in images_b64:
                    img = decode_b64_to_numpy(b64)
                    if img is not None:
                        images.append(img)
                    else:
                        # Create dummy image
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                # Run inference with proper prompt (local retry on empty output)
                prompt = prompt_switch_detection(len(images))
                vlm_json: Dict[str, Any] = {}
                
                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        vlm_json = backend.infer(images, prompt)
                    except Exception as e:
                        print(f"[Err] Inference failed: {e}")
                        vlm_json = {}
                    
                    if not _is_empty_vlm_json(vlm_json):
                        break
                    
                    print(
                        f"[Warn] {task_id} Empty VLM JSON "
                        f"(attempt {attempt + 1}/{MAX_LOCAL_RETRIES})"
                    )
                
                if _is_empty_vlm_json(vlm_json):
                    print(f"[Fail] {task_id} Returning empty to trigger server retry")
                else:
                    print(f"[Done] {task_id} ({len(images)}f) -> Cuts: {vlm_json.get('transitions', [])}")
                
                # Submit result
                requests.post(
                    f"{server_url}/submit_result",
                    json={
                        "task_id": task_id,
                        "vlm_json": vlm_json,
                        "meta": job["meta"]
                    },
                    timeout=30
                )
            
            except KeyboardInterrupt:
                print("[Worker] Stopping...")
                break
            except Exception as e:
                print(f"[Error] Loop crashed: {e}")
                time.sleep(1)
    
    finally:
        backend.cleanup()
