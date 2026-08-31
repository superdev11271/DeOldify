# DeOldify Colorize Server

FastAPI server around the DeOldify ONNX model: send a grayscale image, get a colorized one back.

## Requirements

- Python 3.12
- An ONNX DeOldify model in `models/` (default: `ColorizeArtistic_dyn.onnx`)
- For GPU: NVIDIA driver with CUDA 12 + cuDNN 9

```
pip install -r requirements.txt
```

`requirements.txt` pins `onnxruntime-gpu`. For a CPU-only host, swap it for `onnxruntime`.

## Running

```
python server.py                                  # models/ColorizeArtistic_dyn.onnx on cuda, port 8080
python server.py --device cpu --max_side 1280
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `--model` | `ColorizeArtistic_dyn.onnx` | Model filename inside `models/` |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--max_side` | `1920` | Inputs longer than this on either side are downscaled before inference, then resized back to the original dimensions |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Bind port |

The model loads before the port opens, so startup takes a few seconds; requests during that window get connection errors rather than a 503.

## API

### `GET /api/health`

```json
{"status": "ok", "model": "ColorizeArtistic_dyn.onnx", "device": "cuda", "max_side": 1920}
```

### `POST /api/colorize`

`multipart/form-data`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image` | file | required | Image to colorize |
| `render_factor` | int | `35` | Inference resolution is `render_factor * 16`; higher is slower and resolves finer detail |

Returns the colorized image as `image/png` at the input's original dimensions. `400` if the upload can't be decoded as an image.

```
curl -X POST http://127.0.0.1:8080/api/colorize \
  -F "image=@test/test.png" -F "render_factor=20" -o out.png
```

## Docker

`models/` is excluded from the build context and mounted at run time, so the image stays lean.

```
docker build -t deoldify-server .
docker run --gpus all -p 8080:8080 -v "$PWD/models:/app/models" deoldify-server
```

Server args go after the image name:

```
docker run -p 8080:8080 -v "$PWD/models:/app/models" deoldify-server --device cpu --max_side 1280
```

On Windows, use the absolute path for the mount:

```
docker run --gpus all -p 8080:8080 -v "E:\Project\phoenix\servers\DeOldify\models:/app/models" deoldify-server
```

To bake the model into the image instead, drop `models/` from `.dockerignore` and add `COPY models/ ./models/` to the Dockerfile.

### GPU notes

`--gpus all` requires Docker Desktop's **WSL2 backend** (Settings → General → *Use the WSL 2 based engine*). On the Hyper-V backend the container has no NVIDIA driver and fails with `nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1`.

## CLI

`infer_image.py` colorizes a single file without the server:

```
python infer_image.py --source test/test.png --render_factor 35
```
