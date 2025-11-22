from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import base64
import torch
import uvicorn
import os
import time
from dotenv import load_dotenv
from chess_diagram_to_fen import get_fen

load_dotenv()

app = FastAPI(title="Chess to FEN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DetectRequest(BaseModel):
    img_b64: str
    num_tries: Optional[int] = 15  # Default 3 for speed, can override for accuracy
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
        print(f"\n{'='*60}")
        print(f"🎯 New request - num_tries={request.num_tries}")
        print(f"{'='*60}")

        # 1. Decode base64 image
        decode_start = time.time()
        img_b64 = request.img_b64
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        timings['decode'] = round(time.time() - decode_start, 2)
        print(f"✓ Image decoded: {timings['decode']}s | Size: {img.size}")

        # 2. Run chess detection
        inference_start = time.time()
        print(f"🔍 Starting inference (num_tries={request.num_tries})...")

        result = get_fen(
            img=img,
            num_tries=request.num_tries,
            auto_rotate_image=request.auto_rotate_image,
            auto_rotate_board=request.auto_rotate_board
        )

        timings['inference'] = round(time.time() - inference_start, 2)
        print(f"✓ Inference completed: {timings['inference']}s")

        # 3. Check results
        if result is None or result.fen is None:
            timings['total'] = round(time.time() - total_start, 2)
            print(f"❌ Detection failed - Total time: {timings['total']}s")
            raise HTTPException(status_code=400, detail="Could not detect chess board")

        timings['total'] = round(time.time() - total_start, 2)

        print(f"✅ Success!")
        print(f"   FEN: {result.fen}")
        print(f"   Rotation: {result.image_rotation_angle}°")
        print(f"   Flipped: {result.board_is_flipped}")
        print(f"⏱️  Total time: {timings['total']}s")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "fen": result.fen,
            "rotation_angle": result.image_rotation_angle,
            "is_flipped": result.board_is_flipped,
            "debug": {
                "num_tries": request.num_tries,
                "timings": timings,
                "device": device.type,
                "image_size": f"{img.size[0]}x{img.size[1]}"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        timings['total'] = round(time.time() - total_start, 2)
        print(f"❌ Error: {str(e)}")
        print(f"⏱️  Failed after: {timings['total']}s")
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 80))
    print(f"Device: {device.type}")
    print(f"Server: http://localhost:{PORT}")
    print(f"Docs: http://localhost:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
