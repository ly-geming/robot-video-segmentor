"""Video windowing and frame extraction utilities."""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import base64


@dataclass
class Window:
    """Video window definition."""
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]


def read_video_info(mp4_path: str) -> Tuple[float, int]:
    """Read video FPS and frame count."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps is None or fps != fps or abs(fps) < 1e-6:
        fps = 30.0
    
    return float(fps), max(0, nframes)


def build_windows(
    fps: float,
    nframes: int,
    window_sec: float = 16.0,
    step_sec: float = 8.0,
    frames_per_window: int = 16
) -> List[Window]:
    """Build video windows with frame sampling."""
    if fps < 1e-6:
        fps = 30.0
    
    win_len = max(1, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    windows: List[Window] = []
    
    def get_frames(s: int, e: int, num: int) -> List[int]:
        idx = np.linspace(s, e, num=num).astype(int)
        return np.clip(idx, 0, nframes - 1).tolist()
    
    s = 0
    wid = 0
    while s < nframes:
        e = min(nframes - 1, s + win_len - 1)
        if (e - s < win_len // 2) and wid > 0:
            break
        windows.append(Window(wid, s, e, get_frames(s, e, frames_per_window)))
        wid += 1
        s += step
    
    return windows


def encode_image_720p_png(
    img_bgr: np.ndarray,
    target_w: int = 720,
    target_h: int = 480,
    compression: int = 0
) -> str:
    """Encode image to base64 PNG, resizing if needed."""
    if img_bgr is None:
        return ""
    
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return ""
    
    if (w != target_w) or (h != target_h):
        img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    ok, buf = cv2.imencode(
        ".png",
        img_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(np.clip(compression, 0, 9))]
    )
    
    return base64.b64encode(buf).decode("utf-8") if ok else ""


class FrameExtractor:
    """Extract frames from video file."""
    
    def __init__(self, mp4_path: str):
        self.cap = cv2.VideoCapture(mp4_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")
    
    def close(self) -> None:
        """Release video capture."""
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def get_many_b64(
        self,
        frame_ids: List[int],
        target_w: int = 720,
        target_h: int = 480,
        compression: int = 0
    ) -> List[str]:
        """Extract multiple frames as base64 PNGs."""
        sorted_indices = sorted(list(set(frame_ids)))
        frame_map: dict = {}
        
        for fid in sorted_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, bgr = self.cap.read()
            frame_map[fid] = encode_image_720p_png(
                bgr, target_w, target_h, compression
            ) if (ok and bgr is not None) else ""
        
        return [frame_map.get(fid, "") for fid in frame_ids]


def build_segments_via_cuts(
    sample_id: str,
    windows: List[Window],
    by_wid: dict,
    fps: float,
    nframes: int,
    frames_per_window: int = 16
) -> dict:
    """Build final segments from window results."""
    if nframes == 0:
        return {}
    
    if fps < 1e-6:
        fps = 30.0
    
    from collections import Counter
    
    raw_cuts = []
    instruction_timeline = [[] for _ in range(nframes)]
    center_weights = np.hanning(frames_per_window + 2)[1:-1]
    
    for wid, w in enumerate(windows):
        rec = by_wid.get(wid)
        if not rec:
            continue
        
        vlm = rec.get("vlm_json", {})
        transitions = vlm.get("transitions", [])
        instructions = vlm.get("instructions", [])
        f_ids = w.frame_ids
        cur_len = len(f_ids)
        
        if cur_len == 0:
            continue
        
        # Collect cut points
        for t_idx in transitions:
            try:
                idx = int(t_idx)
                if 0 <= idx < cur_len:
                    global_fid = f_ids[idx]
                    if cur_len == frames_per_window:
                        w_val = center_weights[idx]
                    else:
                        w_val = 1.0 if min(idx, cur_len - 1 - idx) > 2 else 0.5
                    raw_cuts.append((global_fid, float(w_val)))
            except (ValueError, IndexError):
                pass
        
        # Collect instructions
        try:
            boundaries = [0] + [int(t) for t in transitions if 0 <= int(t) < cur_len] + [cur_len]
            boundaries = sorted(list(set(boundaries)))
            
            for i in range(len(boundaries) - 1):
                if i < len(instructions):
                    inst = str(instructions[i]).strip()
                    if inst and inst.lower() != "unknown":
                        s_local, e_local = boundaries[i], boundaries[i + 1]
                        for k in range(s_local, e_local):
                            if k < cur_len:
                                global_fid = f_ids[k]
                                if global_fid < nframes:
                                    instruction_timeline[global_fid].append(inst)
        except (ValueError, IndexError):
            pass
    
    # Cluster cuts
    final_cut_points = [0]
    
    if raw_cuts:
        raw_cuts.sort(key=lambda x: x[0])
        cluster_gap = max(0.5, 0.5 * fps)
        cur_frames = []
        cur_weights = []
        
        for fid, w in raw_cuts:
            if not cur_frames:
                cur_frames.append(fid)
                cur_weights.append(w)
                continue
            
            if (fid - cur_frames[-1]) < cluster_gap:
                cur_frames.append(fid)
                cur_weights.append(w)
            else:
                if cur_weights and sum(cur_weights) > 1e-9:
                    avg = np.average(cur_frames, weights=cur_weights)
                    final_cut_points.append(int(avg))
                else:
                    final_cut_points.append(int(np.mean(cur_frames)))
                cur_frames = [fid]
                cur_weights = [w]
        
        if cur_frames:
            if cur_weights and sum(cur_weights) > 1e-9:
                avg = np.average(cur_frames, weights=cur_weights)
                final_cut_points.append(int(avg))
            else:
                final_cut_points.append(int(np.mean(cur_frames)))
    
    final_cut_points.append(nframes)
    final_cut_points = sorted(list(set(final_cut_points)))
    
    # Build segments
    final_output = []
    seg_id = 0
    
    for i in range(len(final_cut_points) - 1):
        s, e = int(final_cut_points[i]), int(final_cut_points[i + 1])
        min_frames = max(1, int(0.3 * fps))
        
        if (e - s) < min_frames:
            continue
        
        margin = int((e - s) * 0.2) if e > s else 0
        mid_s, mid_e = s + margin, e - margin
        
        candidates = []
        for f in range(mid_s, mid_e + 1):
            if f < nframes:
                candidates.extend(instruction_timeline[f])
        
        if not candidates:
            for f in range(s, e):
                if f < nframes:
                    candidates.extend(instruction_timeline[f])
        
        if candidates:
            best_inst = Counter(candidates).most_common(1)[0][0]
            final_output.append({
                "seg_id": seg_id,
                "start_frame": s,
                "end_frame": e,
                "instruction": best_inst,
                "confidence": 1.0
            })
            seg_id += 1
    
    return {
        "sample_id": sample_id,
        "nframes": nframes,
        "segments": final_output
    }
