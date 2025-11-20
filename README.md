# Chess to FEN API

REST API for converting chess board images to FEN notation using deep learning.

## Features

- Convert chess board images to FEN notation
- Automatic board rotation detection
- Automatic piece color detection
- Base64 image support
- Fast inference with PyTorch

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements-api.txt

# Install package
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
PORT=5000
```

### 3. Run Server

```bash
python app.py
```

Server will run at `http://localhost:5000`

## API Usage

### Endpoint: POST `/api/detect`

Detects chess board from base64 image and returns FEN notation.

**Request:**

```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"img_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUh..."}'
```

**Response:**

```json
{
  "success": true,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "rotation_angle": 0,
  "is_flipped": false
}
```

**Error Response:**

```json
{
  "detail": "Could not detect chess board"
}
```

### Interactive Documentation

Open http://localhost:5000/docs for Swagger UI to test the API directly in your browser.

## Docker

### Build Image

```bash
# Build with no cache (recommended for first build)
docker build --no-cache -t chess-fen-api .

# Or regular build
docker build -t chess-fen-api .
```

### Run Container

```bash
# Run on port 5000
docker run -p 5000:5000 chess-fen-api

# Run with custom port
docker run -p 8000:5000 chess-fen-api

# Run in background
docker run -d -p 5000:5000 --name chess-api chess-fen-api
```

### Troubleshooting Docker

If you get numpy compatibility errors:
```bash
# Rebuild without cache
docker build --no-cache -t chess-fen-api .
```

**Note**: The Dockerfile automatically:
- Installs numpy first to ensure scikit-image compiles correctly
- Pre-downloads PyTorch models during build (regnet_x_800mf & mobilenet_v3_large)
- This prevents slow downloads during runtime/deployment

## Deployment

### Railway / Render / Fly.io

This project is ready for deployment on Railway, Render, or Fly.io:

1. **Connect your repo** to the platform
2. **Set environment variables** (optional):
   - `PORT` - Default: 5000
3. **Deploy** - The Dockerfile will handle everything

**Important**: Models are pre-downloaded during build, so your first deployment will take longer (~5-10 minutes) but subsequent requests will be fast.

### Deployment Platforms

| Platform | Command |
|----------|---------|
| Railway | Connect repo → Deploy automatically |
| Render | Connect repo → Deploy automatically |
| Fly.io | `fly launch` then `fly deploy` |

## Development

### Project Structure

```
.
├── app.py                  # FastAPI application
├── chess_diagram_to_fen.py # Core FEN detection logic
├── models/                 # Pre-trained models
├── src/                    # Model implementations
└── requirements-api.txt    # Python dependencies
```

### Models

This project uses pre-trained models for:
- Chess board detection
- Board orientation detection
- Piece recognition
- FEN generation

Models are located in the `models/` directory.

## License

MIT
