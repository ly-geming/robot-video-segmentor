"""Qwen3-VL backend implementation."""

from typing import List, Dict, Any, Optional
import json
import numpy as np
from PIL import Image
from io import BytesIO
import base64

from .base import VLMBackend


def encode_image_to_pil(img_bgr: np.ndarray, target_w: int = 720, target_h: int = 480) -> Optional[Image.Image]:
    """Convert BGR numpy array to PIL RGB image, resizing if needed."""
    if img_bgr is None:
        return None
    
    import cv2
    h, w = img_bgr.shape[:2]
    
    if h <= 0 or w <= 0:
        return None
    
    if w != target_w or h != target_h:
        img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return {}


def prompt_switch_detection(n_images: int) -> str:
    """Generate prompt for switch detection."""
    return (
        f"You are a vision analyzer for **human egocentric** videos: a {n_images}-frame clip sampled from household manipulation (first-person / ego-view).\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        
        "### Goal\n"
        "Detect **Atomic Task Boundaries** (Switch Points).\n"
        "A 'Switch' occurs strictly when the **person** **completes** interaction with one object and **starts** interacting with a DIFFERENT object.\n\n"
        
        "### Core Logic (The 'Distinct Object' Rule)\n"
        "1. **True Switch:** The person releases Object A (e.g., a cup) and moves to grasp Object B (e.g., a spoon). -> MARK SWITCH.\n"
        "2. **False Switch (IMPORTANT):** If the person is manipulating different parts of the **SAME** object (e.g., folding sleeves then folding the body of the same shirt), this is **NOT** a switch. Treat it as one continuous task.\n"
        "3. **Visual Similarity:** Be careful with objects of the same color. Only mark a switch if you clearly see the person **physically separate** from the first item before touching the second.\n\n"
        
        "### Output Format: Strict JSON\n"
        "Your response must be a valid JSON object including a 'thought' field for step-by-step analysis, 'transitions' for the switch indices, and 'instructions' for the task labels.\n\n"
        
        "### Representative Examples\n"
        "**Example 1: Table Setting (True Switch)**\n"
        "{\n"
        '  "thought": "Frames 0-5: The person places a fork. Frame 6: Hand releases fork and moves to the spoon. Frame 7: Hand grasps spoon. Switch detected at 6.",\n'
        '  "transitions": [6],\n'
        '  "instructions": ["Place the fork", "Place the spoon"]\n'
        "}\n\n"
        
        "**Example 2: Folding Laundry (False Switch - Same Object)**\n"
        "{\n"
        '  "thought": "Frames 0-10: The person folds the left sleeve of the black shirt. Frames 11-20: The person folds the body of the **same** black shirt. Although the grasp changed, the object remains the same. The action is continuous.",\n'
        '  "transitions": [],\n'
        '  "instructions": ["Fold the black shirt"]\n'
        "}\n\n"
        
        "**Example 3: Cleaning (Continuous)**\n"
        "{\n"
        '  "thought": "Frames 0-15: The person is wiping the counter. The motion is repetitive, but it is the same task. No switch.",\n'
        '  "transitions": [],\n'
        '  "instructions": ["Wipe the counter"]\n'
        "}"
    )


class Qwen3VLBackend(VLMBackend):
    """Qwen3-VL backend using Transformers."""
    
    def __init__(self, model_path: str, device_map: str = "balanced"):
        self.model_path = model_path
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.target_w = 720
        self.target_h = 480
    
    @property
    def name(self) -> str:
        return "qwen3vl"
    
    def warmup(self) -> None:
        """Load model and processor."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        
        print(f"[Qwen3VL] Loading model from {self.model_path}...")
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device_map
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.eval()
        
        print("[Qwen3VL] Model ready.")
    
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Run inference with Qwen3-VL."""
        import torch
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call warmup() first.")
        
        # Convert images to PIL
        pil_images = []
        for img in images:
            pil_img = encode_image_to_pil(img, self.target_w, self.target_h)
            if pil_img is None:
                pil_img = Image.new('RGB', (224, 224))
            pil_images.append(pil_img)
        
        # Use provided prompt (already generated by caller)
        full_prompt = prompt
        
        # Prepare inputs
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images] + 
                       [{"type": "text", "text": full_prompt}]
        }]
        
        text_inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_inputs], 
            images=pil_images, 
            padding=True, 
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Extract JSON (return empty dict on parse failure)
        result = extract_json(output_text)
        return result
    
    def cleanup(self) -> None:
        """Cleanup model resources."""
        import torch
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        torch.cuda.empty_cache()
        print("[Qwen3VL] Model cleaned up.")
