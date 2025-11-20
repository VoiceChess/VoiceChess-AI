from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch
import uvicorn
import os
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

@app.post("/api/detect")
async def detect(request: DetectRequest):
    """
    Detect chess board from base64 image and return FEN notation
    """
    try:
        # Remove data URL prefix if present
        img_b64 = request.img_b64
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        # Decode base64 to image
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))

        # Get FEN from image
        result = get_fen(img=img, num_tries=10, auto_rotate_image=True, auto_rotate_board=True)

        if result is None or result.fen is None:
            raise HTTPException(status_code=400, detail="Could not detect chess board")

        return {
            "success": True,
            "fen": result.fen,
            "rotation_angle": result.image_rotation_angle,
            "is_flipped": result.board_is_flipped
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 5000))
    print(f"Device: {device.type}")
    print(f"Server: http://localhost:{PORT}")
    print(f"Docs: http://localhost:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
