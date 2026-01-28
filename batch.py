#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch meme sentiment flip using Qwen-Image-Edit (OpenAI-compatible chat endpoint),
with customizable batch scheduling + concurrent workers.

Inputs:
  - A JSON/JSONL metadata file with records like:
    {
      "id": "0.png",
      "spec": "...",
      "tgt_emo": "calm",
      "pos_txt": "..."
    }

Images:
  - Located under: /home/kun.li/meme/dataset/Original/{id}  (default)

Outputs:
  - Saved under: {out_root}/{model_name}/{id}
  - The folder name is exactly model_name (no /v1/models call, since your server's /v1/models is broken)

Server:
  - You said you launched:
    CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen-Image-Edit-2509 --omni --port 8092 --usp 2 --gpu_memory_utilization 0.95

Example:
  python batch.py \
    --json index_filtered.json \
    --image_root dataset/Original \
    --server http://localhost:8091 \
    --out_root Generated \
    --model_name FLUX.2-klein-4B \
    --batch_size 2 --workers 2 \
    --steps 50 --guidance 7.5 --seed -1 \
    --skip_existing
"""

import argparse
import base64
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

# 2
PROMPT_TEMPLATE = """[Task] Perform a holistic sentiment flip on the entire image from negative to positive, guided by the instruction: "{spec}".

[Constraints]

PRESERVE STRICTLY: Character identity, features, style, panel layout, background objects, and the original meme template structure.

MODIFY GLOBALLY: All expressions and overall emotion. Text captions must be replaced.

[Execution Steps]

VISUAL: Transform the character's expression and overall emotion to genuinely reflect a "{tgt_emo}" state. The transition must feel natural and relaxed.

TEXT: Replace existing text overlay with the exact phrase: "{pos_txt}".
"""

# 1
# PROMPT_TEMPLATE = """[Task] Perform a holistic sentiment flip on the entire image from negative to positive, guided by the instruction: "{spec}".

# [Constraints]

# PRESERVE STRICTLY: Character identity, clothing features, panel layout, background objects, and the original meme template structure.

# MODIFY GLOBALLY: All expressions and overall emotion. Text captions must be replaced.

# [Execution Steps]

# VISUAL: Transform the character's expression and overall emotion to genuinely reflect a "{tgt_emo}" state. The transition must feel natural and relaxed.

# TEXT: Replace existing text overlay with the exact phrase: "{pos_txt}".
# """

# PROMPT_TEMPLATE = """[Task] Perform a holistic sentiment flip on the entire image from negative to positive, guided by the instruction: "{spec}".

# [Constraints]

# PRESERVE STRICTLY: Character identity, clothing features, panel layout, background objects, and the original meme template structure.

# MODIFY GLOBALLY: All expressions and overall atmospheric mood. Text captions must be replaced.

# [Execution Steps]

# VISUAL: Transform the character's facial expression to genuinely reflect a "{tgt_emo}" state. The transition must feel natural and relaxed. Remove dark, heavy, or high-contrast tension.

# TEXT: Replace existing text overlay with the exact phrase: "{pos_txt}".
# """


def _encode_image_as_data_url(input_path: Path) -> str:
    image_bytes = input_path.read_bytes()
    try:
        img = Image.open(BytesIO(image_bytes))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
    except Exception:
        mime_type = "image/png"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def extract_first_data_image_url(content_obj: Any) -> Optional[str]:
    """
    Find first 'data:image...' url from OpenAI-style multimodal message content.
    content_obj can be:
      - string
      - list of parts: [{"type":"image_url","image_url":{"url":"data:image..."}}, ...]
    """
    if isinstance(content_obj, list):
        for part in content_obj:
            if not isinstance(part, dict):
                continue
            img_url = (part.get("image_url") or {}).get("url")
            if isinstance(img_url, str) and img_url.startswith("data:image"):
                return img_url
    return None


def edit_image(
    input_image: Union[str, Path, List[Union[str, Path]]],
    prompt: str,
    server_url: str,
    model_name: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    timeout: int = 300,
) -> Optional[bytes]:
    """
    Send an OpenAI-compatible /v1/chat/completions request to edit one image.
    Returns edited image bytes, or None on failure.

    Note: Do NOT share requests.Session across threads. We create a new Session per call.
    """
    input_images = input_image if isinstance(input_image, list) else [input_image]
    input_paths = [Path(p) for p in input_images]
    for p in input_paths:
        if not p.exists():
            print(f"[Error] Input image not found: {p}")
            return None

    # Build user message with text and image(s)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in input_paths:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_as_data_url(p)}})

    messages = [{"role": "user", "content": content}]

    extra_body: Dict[str, Any] = {}
    if steps is not None:
        extra_body["num_inference_steps"] = steps
    if guidance_scale is not None:
        extra_body["guidance_scale"] = guidance_scale
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt
    if height is not None:
        extra_body["height"] = height
    if width is not None:
        extra_body["width"] = width

    payload: Dict[str, Any] = {
        "model": model_name,    # IMPORTANT: specify model explicitly
        "messages": messages,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    url = f"{server_url.rstrip('/')}/v1/chat/completions"

    try:
        with requests.Session() as s:
            response = s.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )

        # If server returned error JSON, expose it
        if response.status_code != 200:
            try:
                print(f"[Error] HTTP {response.status_code}: {response.json()}")
            except Exception:
                print(f"[Error] HTTP {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        data = response.json()
        msg = data["choices"][0]["message"]
        content_obj = msg.get("content")

        image_url = extract_first_data_image_url(content_obj)
        if image_url:
            _, b64_data = image_url.split(",", 1)
            return base64.b64decode(b64_data)

        print(f"[Warn] Unexpected response content format: {content_obj}")
        return None

    except Exception as e:
        print(f"[Error] Request failed: {e}")
        return None


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSON array file: [ {...}, {...} ]
      - JSONL: each line is a JSON object
    """
    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError("JSON starts with '[' but is not a list.")
        return [x for x in obj if isinstance(x, dict)]

    # jsonl
    records: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _validate_record(rec: Dict[str, Any]) -> bool:
    """Minimal required fields for your prompt template."""
    img_id = rec.get("id")
    return (
        isinstance(img_id, str)
        and rec.get("spec") is not None
        and rec.get("tgt_emo") is not None
        and rec.get("pos_txt") is not None
    )


def process_one(
    rec: Dict[str, Any],
    image_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> str:
    """
    Process a single record with retry.
    Returns: "success" | "failed" | "skipped"
    """
    if not _validate_record(rec):
        return "skipped"

    img_id = rec["id"]
    in_path = image_root / img_id
    if not in_path.exists():
        return "skipped"

    out_path = out_dir / img_id
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    prompt = PROMPT_TEMPLATE.format(
        spec=str(rec["spec"]),
        tgt_emo=str(rec["tgt_emo"]),
        pos_txt=str(rec["pos_txt"]),
    )

    seed = random.randint(0, 2**31 - 1) if args.seed == -1 else int(args.seed)

    for attempt in range(args.retries + 1):
        image_bytes = edit_image(
            input_image=str(in_path),
            prompt=prompt,
            server_url=args.server,
            model_name=args.model_name,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed,
            negative_prompt=args.negative,
            timeout=args.timeout,
        )

        if image_bytes:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image_bytes)
            return "success"

        if attempt < args.retries:
            time.sleep(args.retry_backoff * (attempt + 1))

    return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Batch Qwen-Image-Edit meme sentiment flip client with custom batch+concurrency."
    )
    parser.add_argument("--json", required=True, help="Path to JSON/JSONL metadata file")
    parser.add_argument("--image_root", default="/home/kun.li/meme/dataset/Original", help="Root dir of original images")
    parser.add_argument("--server", default="http://localhost:8092", help="Server URL, e.g., http://localhost:8092")

    # IMPORTANT: /v1/models is broken on your server; we use model_name directly.
    parser.add_argument("--model_name", default="Qwen-Image-Edit-2509", help="Served model name (also output folder).")
    parser.add_argument("--out_root", default=".", help="Root output dir; model folder created inside it")

    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Seed. -1 => random per sample.")
    parser.add_argument("--negative", default=None, help="Negative prompt")
    parser.add_argument("--timeout", type=int, default=99999999, help="HTTP timeout (seconds) per request")

    # Batch scheduling + concurrency
    parser.add_argument("--batch_size", type=int, default=8, help="How many samples to schedule per batch loop.")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent requests (threads).")
    parser.add_argument("--retries", type=int, default=2, help="Retry times for failed requests.")
    parser.add_argument("--retry_backoff", type=float, default=1.5, help="Backoff factor (seconds) between retries.")

    parser.add_argument("--skip_existing", action="store_true", help="Skip if output file already exists")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples (0 = no limit)")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    out_dir = Path(args.out_root) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(json_path)
    if not records:
        print("[Error] No records loaded from JSON/JSONL.")
        return

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    total = len(records)

    print(f"Server:     {args.server}")
    print(f"Model name: {args.model_name}")
    print(f"Image root: {image_root}")
    print(f"Out dir:    {out_dir}")
    print(f"Records:    {total}")
    print(f"batch_size: {args.batch_size}, workers: {args.workers}, retries: {args.retries}\n")

    processed = 0
    success = 0
    failed = 0
    skipped = 0

    # Schedule in batches (batch_size) with a thread pool (workers)
    for start in range(0, total, args.batch_size):
        batch = records[start: start + args.batch_size]
        if not batch:
            continue

        print(f"[Batch] {start + 1}-{min(start + len(batch), total)} / {total}")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, rec, image_root, out_dir, args) for rec in batch]
            for fut in as_completed(futures):
                status = fut.result()
                processed += 1
                if status == "success":
                    success += 1
                elif status == "failed":
                    failed += 1
                else:
                    skipped += 1

        print(f"[Batch Done] processed={processed} success={success} failed={failed} skipped={skipped}\n")

    print("==== Summary ====")
    print(f"Processed: {processed}")
    print(f"Success:   {success}")
    print(f"Failed:    {failed}")
    print(f"Skipped:   {skipped}")
    print(f"Output:    {out_dir}")


if __name__ == "__main__":
    main()
