"""FastAPI server exposing DeOldify ONNX colorization at POST /api/colorize.

Start with a model name and a device; the models always live in `MODEL_DIR`:

    python server.py --model ColorizeArtistic_dyn.onnx --device cuda

POST an image (multipart field `image`) and get the colorized image back as
PNG. Images whose longest side exceeds `--max_side` are downscaled for
inference and scaled back up to their original size. GET /api/health for a
liveness check.
"""
import argparse
import os

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from deoldify_onnx import DEOLDIFY

MODEL_DIR = 'models'
DEFAULT_MODEL = 'ColorizeArtistic_dyn.onnx'
DEFAULT_DEVICE = 'cuda'
DEFAULT_MAX_SIDE = 1920
DEFAULT_RENDER_FACTOR = 35

app = FastAPI(title='DeOldify ONNX')
colorizer = None
model_info = {}
max_side = DEFAULT_MAX_SIDE


@app.get('/api/health')
async def health():
    """Report whether the model is loaded and how it is configured."""
    if colorizer is None:
        raise HTTPException(status_code=503, detail='model not loaded')
    return {'status': 'ok', **model_info, 'providers': colorizer.session.get_providers()}


@app.post('/api/colorize')
async def colorize(image: UploadFile = File(...), render_factor: int = Form(DEFAULT_RENDER_FACTOR)):
    """Colorize the uploaded image and return it as PNG."""
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='could not decode image')

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        # shrink the long side to max_side, colorize, then come back to the original size
        ratio = max_side / max(h, w)
        small = cv2.resize(img, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
        output = cv2.resize(colorizer.colorize(small, render_factor), (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        output = colorizer.colorize(img, render_factor)
    ok, buf = cv2.imencode('.png', output)
    if not ok:
        raise HTTPException(status_code=500, detail='could not encode result')
    return Response(content=buf.tobytes(), media_type='image/png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL,
                        help=f'model file name inside {MODEL_DIR}/')
    parser.add_argument('-d', '--device', type=str, default=DEFAULT_DEVICE, choices=['cuda', 'cpu'])
    parser.add_argument('--max_side', type=int, default=DEFAULT_MAX_SIDE,
                        help='images with a longer side than this are downscaled before inference')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=8080)
    args = parser.parse_args()

    global colorizer, max_side
    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    colorizer = DEOLDIFY(model_path=model_path, device=args.device)
    max_side = args.max_side
    model_info.update(model=args.model, device=args.device, max_side=args.max_side)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
