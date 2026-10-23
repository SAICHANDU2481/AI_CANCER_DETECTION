"""
================================================================================
V8 Phase 1: High-Resolution VRAM & GPU Inference Benchmark
================================================================================
Benchmarks U-KAN V6 inference across resolutions:
- 512x512
- 768x768
- 1024x1024
Measures: peak allocated VRAM, peak reserved VRAM, runtime/image, AMP status.
Saves: results/v8/phase1/phase1_vram_benchmark.json
================================================================================
"""

import os
import sys
import time
import json
import numpy as np
import torch

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segmentation.v6.ukan_v6 import MultiScaleUKANV6
from app.config import UKAN_V6_PATH, EXPECTED_MD5, verify_file_md5

def run_vram_benchmark():
    print("=" * 80)
    print("V8 PHASE 1: HIGH-RESOLUTION VRAM & INFERENCE BENCHMARK")
    print("=" * 80)

    is_ok, md5 = verify_file_md5(UKAN_V6_PATH, EXPECTED_MD5['ukan_v6'])
    assert is_ok, f"U-KAN checkpoint hash mismatch: {md5}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Model
    ukan = MultiScaleUKANV6(in_channels=1, out_channels=1)
    ckpt = torch.load(UKAN_V6_PATH, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    ukan.load_state_dict(sd)
    ukan.to(device).eval()

    resolutions = [(512, 512), (768, 768), (1024, 1024)]
    results = {}

    for res in resolutions:
        H, W = res
        res_str = f"{H}x{W}"
        print(f"\nEvaluating resolution {res_str}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)

        # Warmup
        dummy_in = torch.randn(1, 1, H, W, device=device, dtype=torch.float32)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                _ = ukan(dummy_in)

        # Timed runs
        times = []
        n_iters = 10
        try:
            for _ in range(n_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        out = ukan(dummy_in)
                        prob = torch.sigmoid(out)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - t0)

            mean_time_ms = float(np.mean(times) * 1000.0)
            std_time_ms = float(np.std(times) * 1000.0)

            alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
            resv_mb = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0

            status = "SUPPORTED_SAFE" if alloc_mb < 3200 else "HIGH_VRAM_WARNING"
            print(f"   [OK] {res_str}: Latency = {mean_time_ms:.1f} +/- {std_time_ms:.1f} ms | Allocated VRAM = {alloc_mb:.1f} MB | Reserved = {resv_mb:.1f} MB | Status: {status}")

            results[res_str] = {
                "resolution": [H, W],
                "mean_latency_ms": round(mean_time_ms, 2),
                "std_latency_ms": round(std_time_ms, 2),
                "peak_allocated_vram_mb": alloc_mb,
                "peak_reserved_vram_mb": resv_mb,
                "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "batch_size": 1,
                "amp_fp16": True,
                "status": status,
                "error": None
            }

        except torch.cuda.OutOfMemoryError as oom:
            print(f"   [OOM] {res_str}: Out of Memory! Error: {oom}")
            results[res_str] = {
                "resolution": [H, W],
                "mean_latency_ms": None,
                "std_latency_ms": None,
                "peak_allocated_vram_mb": None,
                "peak_reserved_vram_mb": None,
                "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "batch_size": 1,
                "amp_fp16": True,
                "status": "UNSUPPORTED_OOM",
                "error": str(oom)
            }
            torch.cuda.empty_cache()

    out_json = os.path.join("results", "v8", "phase1", "phase1_vram_benchmark.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Saved VRAM benchmark report to {out_json}")

if __name__ == "__main__":
    run_vram_benchmark()
