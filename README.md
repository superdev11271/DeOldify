# DeOldify ONNX

Colorize grayscale images with an exported DeOldify ONNX model -- as a Python
class and a FastAPI server. Send a grayscale image, get a colorized one back.

## Layout

| File | Purpose |
| --- | --- |
| [deoldify_onnx.py](deoldify_onnx.py) | `DEOLDIFY` -- the colorizer (preprocess -> session -> postprocess) |
| [server.py](server.py) | FastAPI server |
| [test.py](test.py) | Smoke-tests the API against an image you pass in |
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

colorizer = DEOLDIFY('models/ColorizeArtistic_dyn_fp16.onnx', device='cuda')
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
python server.py -m ColorizeArtistic_dyn.onnx -d cpu -p 9000
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m`, `--model` | `ColorizeArtistic_dyn_fp16.onnx` | Colorization model file name |
| `-d`, `--device` | `cuda` | `cuda` or `cpu` |
| `--max_side` | `1280` | Images with a longer side than this are downscaled before inference |
| `--host` | `0.0.0.0` | Bind address |
| `-p`, `--port` | `8080` | Bind port |

The model directory is always `models/`, so `--model` take a file name, not a path.
Models load once at startup and a missing file fails immediately, so startup takes
a few seconds and the port only opens once every session is ready.

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
{"status": "ok", "model": "ColorizeArtistic_dyn_fp16.onnx", "device": "cuda", "max_side": 1280,
 "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

`providers` comes from the live session, so it shows whether CUDA actually engaged or
fell back to CPU. `503` before loading has finished.

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
curl -X POST -F "image=@test/test.png" http://127.0.0.1:8080/api/colorize -o test/test_out.png
```

## Docker

Base image `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`, so run it with `--gpus all`.
`models/` is not baked into the image (~370 MB of onnx) -- mount it at run time.

```bash
docker build -t deoldify-server .
docker run --gpus all -p 8080:8080 -v ./models:/app/models deoldify-server
```

Server flags pass straight through the entrypoint:

```bash
docker run --gpus all -p 8080:8080 -v ./models:/app/models deoldify-server -m ColorizeArtistic_dyn.onnx --max_side 1920
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

`test.py` takes the input image path as its argument and checks the health route, a
round-trip of that image, and that a non-image upload is rejected. It exits non-zero
if any check fails. Each line carries the response time for that request, so you can
see what the model actually costs per call. It needs `requests`, which is deliberately
kept out of `requirements.txt` so it stays out of the Docker image.

```bash
pip install requests
python server.py --device cpu &
python test.py test/test.png
```

```
[PASS] health -- 5ms -- {'status': 'ok', 'model': 'ColorizeArtistic_dyn_fp16.onnx', ...}
[PASS] colorize -- 6057ms -- (763, 596, 3) -> (763, 596, 3)
       wrote test/test_out.png
[PASS] rejects a non-image -- 5ms -- 400 {"detail":"could not decode image"}
3/3 checks passed
```

Those times are from a CPU run on this sample; `--device cuda` and the fp16 graphs
give very different numbers.

| Flag | Default | Meaning |
| --- | --- | --- |
| `image` | required | Path to the image to send |
| `--url` | `http://127.0.0.1:8080` | Base url of the running server |
| `-o`, `--output` | `test/test_out.png` | Where to write the returned png; defaults to the input path with `_out.png` in place of its extension |
| `--render_factor` | `35` | Inference resolution is `render_factor * 16` |

Point it at another host or a container with `--url`:

```bash
python test.py test/test.png --url http://127.0.0.1:9000
```

### With curl

The same three checks by hand, without `test.py` or `requests`:

```bash
curl -s http://127.0.0.1:8080/api/health
# {"status": "ok", ...}

curl -s -X POST -F "image=@test/test.png" http://127.0.0.1:8080/api/colorize -o test/test_out.png
python -c "import cv2; print(cv2.imread('test/test.png').shape, '->', cv2.imread('test/test_out.png').shape)"
# (763, 596, 3) -> (763, 596, 3)

curl -s -o /dev/null -w '%{http_code}\n' -X POST -F "image=@README.md" http://127.0.0.1:8080/api/colorize
# 400
```

## Notes

- ONNX sessions are created at startup and shared. FastAPI runs the endpoint in a
  threadpool, so concurrent requests are correct but throughput is bounded by the
  session a request runs on.
- The default is the fp16 graph. The fp32 one is selected by name:
  `--model ColorizeArtistic_dyn.onnx`.
