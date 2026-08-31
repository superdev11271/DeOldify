# DeOldify ONNX

Colorize grayscale images with an exported DeOldify ONNX model -- as a Python
class and a FastAPI server. Send a grayscale image, get a colorized one back.

## Layout

| File | Purpose |
| --- | --- |
| [deoldify_onnx.py](deoldify_onnx.py) | `DEOLDIFY` -- the colorizer (preprocess -> session -> postprocess) |
| [server.py](server.py) | FastAPI server |
| `models/` | `.onnx` weights (not tracked in git) |
| `test/` | Sample images used by the smoke checks below (not tracked in git) |

## Install

```bash
pip install -r requirements.txt
```

Python 3.12. `requirements.txt` pins `onnxruntime-gpu`, which needs an NVIDIA driver
with CUDA 12 + cuDNN 9. For a CPU-only machine, install `onnxruntime` instead and
start the server with `--device cpu`.

## Library

```python
import cv2
from deoldify_onnx import DEOLDIFY

colorizer = DEOLDIFY('models/ColorizeArtistic_dyn.onnx', device='cuda')
output = colorizer.colorize(cv2.imread('input.png'), 35)     # BGR uint8 in, BGR uint8 out
outputs = colorizer.colorize_batch([img1, img2], 35)         # needs a model exported with --dynamic
```

- fp32 and fp16 models are both supported; the input dtype is read from the model.
- `device` is `'cuda'` or `'cpu'` (default); `'cuda'` falls back to CPU when no CUDA
  execution provider is available.
- Inference runs at `render_factor * 16` square. The chroma comes from the model
  output and the luma from your source image, so the result keeps the input's detail.

## Server

```bash
python server.py
python server.py -m ColorizeStable_dyn_fp16.onnx -d cpu -p 9000
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m`, `--model` | `ColorizeArtistic_dyn.onnx` | Colorization model file name |
| `-d`, `--device` | `cuda` | `cuda` or `cpu` |
| `--max_side` | `1920` | Images with a longer side than this are downscaled before inference |
| `--host` | `0.0.0.0` | Bind address |
| `-p`, `--port` | `8080` | Bind port |

The model directory is always `models/`, so `--model` take a file name, not a path.
The model is loaded once at startup and a missing file fails immediately, so startup
takes a few seconds and the port only opens once the session is ready.

An upload whose longer side exceeds `--max_side` is downscaled to that limit (aspect
ratio kept), colorized, then resized back to its **original** dimensions, so oversized images
come back the same size they went in.

## API

Interactive docs at `/docs`.

### `GET /api/health`

```bash
curl http://127.0.0.1:8080/api/health
```

```json
{"status": "ok", "model": "ColorizeArtistic_dyn.onnx", "device": "cuda", "max_side": 1920,
 "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

`providers` comes from the live session, so it shows whether CUDA actually engaged or
fell back to CPU. `503` if the model is not loaded.

### `POST /api/colorize`

Request -- `multipart/form-data`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image` | file | required | Image to colorize |
| `render_factor` | int | `35` | Inference resolution is `render_factor * 16`; higher is slower and resolves finer detail |

Response -- the colorized image as raw `image/png` bytes at the input's original
dimensions. `400` if the upload cannot be decoded as an image, `500` if the result
cannot be encoded.

```bash
curl -X POST -F "image=@test/test.png" http://127.0.0.1:8080/api/colorize -o out.png
```

## Docker

Base image `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`, so run it with `--gpus all`.
`models/` is not baked into the image (~2.4 GB of onnx) -- mount it at run time.

```bash
docker build -t deoldify-server .
docker run --gpus all -p 8080:8080 -v ./models:/app/models deoldify-server
```

Server flags pass straight through the entrypoint:

```bash
docker run --gpus all -p 8080:8080 -v ./models:/app/models deoldify-server -m ColorizeStable_dyn_fp16.onnx --max_side 1280
```

Omit `--gpus all` and pass `-d cpu` to run on CPU. On Windows use an absolute path for
the mount, e.g. `-v "E:\Project\phoenix\servers\DeOldify\models:/app/models"`.

To bake the models into the image instead, drop `models/` from `.dockerignore` and add
`COPY models/*.onnx ./models/` to the Dockerfile.

### GPU notes

`--gpus all` requires Docker Desktop's **WSL2 backend** (Settings -> General -> *Use the
WSL 2 based engine*). On the Hyper-V backend the container has no NVIDIA driver and
fails with `nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1`.

## Test

Start the server, then check that it is up, that a real image round-trips at its
original size, and that a non-image is rejected.

```bash
python server.py --device cpu &

curl -s http://127.0.0.1:8080/api/health
# {"status": "ok", ...}

curl -s -X POST -F "image=@test/test.png" http://127.0.0.1:8080/api/colorize -o out.png
python -c "import cv2; print(cv2.imread('test/test.png').shape, '->', cv2.imread('out.png').shape)"
# the two shapes match

curl -s -o /dev/null -w '%{http_code}\n' -X POST -F "image=@README.md" http://127.0.0.1:8080/api/colorize
# 400
```

## Notes

- One ONNX session is created at startup and shared. FastAPI runs the endpoint in a
  threadpool, so concurrent requests are correct but throughput is bounded by the
  single session.
- The `_fp16` graphs are selected by name, e.g. `--model ColorizeStable_dyn_fp16.onnx`.
- `models/ColorizeStable_gen.pth` is the original torch checkpoint the ONNX graph was
  exported from. It is unused at inference time and excluded from the Docker image.
