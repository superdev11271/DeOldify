import os
from argparse import ArgumentParser

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from deoldify_onnx import DEOLDIFY

MODEL_DIR = "models"
DEFAULT_MODEL = "ColorizeArtistic_dyn.onnx"
DEFAULT_DEVICE = "cuda"
DEFAULT_RENDER_FACTOR = 35
# images larger than this on either side are downscaled before inference
DEFAULT_MAX_SIDE = 1920

app = FastAPI(title="DeOldify colorize server")
colorizer = None
max_side = DEFAULT_MAX_SIDE
model_info = {}


@app.get("/api/health")
async def health():
    return {"status": "ok" if colorizer is not None else "loading", **model_info}


@app.post("/api/colorize")
async def colorize(image: UploadFile = File(...), render_factor: int = Form(DEFAULT_RENDER_FACTOR)):
    data = np.frombuffer(await image.read(), np.uint8)
    source = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if source is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    h, w = source.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        resized = cv2.resize(source, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        colorized = colorizer.colorize(resized, render_factor)
        colorized = cv2.resize(colorized, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        colorized = colorizer.colorize(source, render_factor)

    ok, encoded = cv2.imencode(".png", colorized)
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode result")
    return Response(content=encoded.tobytes(), media_type="image/png")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="onnx model name inside the models dir")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda or cpu")
    parser.add_argument("--max_side", type=int, default=DEFAULT_MAX_SIDE,
                        help="images larger than this on either side are downscaled before inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    opt = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, opt.model)

    global colorizer, max_side
    max_side = opt.max_side
    colorizer = DEOLDIFY(model_path=model_path, device=opt.device)
    model_info.update(model=opt.model, device=opt.device, max_side=opt.max_side)

    uvicorn.run(app, host=opt.host, port=opt.port)


if __name__ == '__main__':
    main()
