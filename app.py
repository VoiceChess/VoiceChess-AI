import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import base64
import io
import json
import logging
import time
from typing import Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from chess_diagram_to_fen import get_fen

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voicechess.fen")


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))


app = FastAPI(title="Chess to FEN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error=str(error),
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    log_event(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.on_event("startup")
async def startup() -> None:
    log_event(
        "startup",
        device=device.type,
        port=os.getenv("PORT", "5000"),
        torch_num_threads=torch.get_num_threads(),
    )


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "VoiceChessAI", "device": device.type}


class DetectRequest(BaseModel):
    img_b64: str
    num_tries: Optional[int] = 5  # Default 3 for speed, can override for accuracy
    auto_rotate_image: Optional[bool] = True
    auto_rotate_board: Optional[bool] = True


@app.post("/api/detect")
async def detect(request: DetectRequest):
    """
    Detect chess board from base64 image and return FEN notation

    Parameters:
    - img_b64: Base64 encoded image string
    - num_tries: Number of inference attempts (default: 3)
      * 1-2: Fast but less accurate (~2-4s)
      * 3-5: Balanced (recommended) (~5-10s)
      * 10+: High accuracy but slow (~20-30s)
    - auto_rotate_image: Auto-detect image rotation (default: True)
    - auto_rotate_board: Auto-detect board perspective (default: True)
    """
    timings = {}
    total_start = time.time()

    try:
        request.num_tries = max(1, min(request.num_tries or 3, 5))
        log_event(
            "fen_detect_started",
            num_tries=request.num_tries,
            auto_rotate_image=request.auto_rotate_image,
            auto_rotate_board=request.auto_rotate_board,
        )

        decode_start = time.time()
        img_b64 = request.img_b64
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img.thumbnail((1200, 1200))
        timings["decode"] = round(time.time() - decode_start, 2)
        log_event(
            "fen_image_decoded",
            decode_ms=round(timings["decode"] * 1000, 2),
            width=img.size[0],
            height=img.size[1],
        )

        inference_start = time.time()
        log_event("fen_inference_started", num_tries=request.num_tries)

        result = get_fen(
            img=img,
            num_tries=request.num_tries,
            auto_rotate_image=request.auto_rotate_image,
            auto_rotate_board=request.auto_rotate_board,
        )

        timings["inference"] = round(time.time() - inference_start, 2)
        log_event(
            "fen_inference_completed",
            inference_ms=round(timings["inference"] * 1000, 2),
        )

        if result is None or result.fen is None:
            timings["total"] = round(time.time() - total_start, 2)
            log_event(
                "fen_detect_failed",
                reason="no_board_detected",
                total_ms=round(timings["total"] * 1000, 2),
            )
            raise HTTPException(status_code=400, detail="Could not detect chess board")

        timings["total"] = round(time.time() - total_start, 2)

        log_event(
            "fen_detect_completed",
            fen=result.fen,
            rotation_angle=result.image_rotation_angle,
            is_flipped=result.board_is_flipped,
            total_ms=round(timings["total"] * 1000, 2),
            timings=timings,
        )

        return {
            "success": True,
            "fen": result.fen,
            "rotation_angle": result.image_rotation_angle,
            "is_flipped": result.board_is_flipped,
            "debug": {
                "num_tries": request.num_tries,
                "timings": timings,
                "device": device.type,
                "image_size": f"{img.size[0]}x{img.size[1]}",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        timings["total"] = round(time.time() - total_start, 2)
        log_event(
            "fen_detect_failed",
            error_type=type(e).__name__,
            error=str(e),
            total_ms=round(timings["total"] * 1000, 2),
        )
        raise HTTPException(status_code=500, detail="Chess board analysis failed")


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 80))
    log_event("server_start", device=device.type, port=PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
