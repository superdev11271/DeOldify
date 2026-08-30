# deoldify-onnx

Updated version: option render factor added (only commandline version)

New models for use with render factor: 

https://drive.google.com/drive/folders/1bU9Zj7zGVEujIzvDTb1b9cyWU3s__WQR?usp=sharing

.

Simple image and video colorization using onnx converted deoldify model.

Easy to install. Can be run on CPU or nVidia GPU

ffmpeg for video colorzation required.

Added floating point 16 model for 100% faster inference and simple GUI version.

## fp16 vs fp32 benchmark

ColorizeArtistic_dyn, raw onnxruntime session.run (no pre-/post-processing),
batch 1, RTX 4070, onnxruntime-gpu 1.26.0, CUDA EP, 5 warmup + 15-20 timed runs.

With cudnn_conv_algo_search "DEFAULT":

| render_factor | resolution | fp32 | fp16 | speedup |
|---|---|---|---|---|
| 21 | 336x336 | 74.59 ms | 74.46 ms | 1.00x |
| 35 | 560x560 | 193.11 ms | 184.78 ms | 1.05x |
| 45 | 720x720 | 315.43 ms | 300.04 ms | 1.05x |

fp16 gives almost nothing here because the cuDNN heuristic picks an NCHW
convolution algorithm that does not use tensor cores. Letting cuDNN autotune
unlocks the real gain, at 560x560:

| cudnn_conv_algo_search | fp32 | fp16 | speedup |
|---|---|---|---|
| DEFAULT | 193.11 ms | 184.78 ms | 1.05x |
| EXHAUSTIVE | 118.60 ms | 70.24 ms | 1.69x |

So fp16 is ~1.7x faster than fp32, and fp16 + EXHAUSTIVE is ~2.75x faster than
the fp32/DEFAULT path. EXHAUSTIVE costs a one-time autotune per input shape on
the first run. The setting lives in deoldify_onnx.py.

fp16 also halves the model file (128 MB vs 255 MB) and loads about twice as
fast (0.55 s vs 1.2 s).

For inference run:

Image:
python infer_image.py --source "image.jpg"

Video:
python infer_video.py --source "video.mp4" --result "video_colorized.mp4" --audio

Image example:
![colorizer1](https://github.com/instant-high/deoldify-onnx/assets/77229558/171642dd-9034-4ca7-8d29-c07c6e5e9f0a)


https://github.com/instant-high/deoldify-onnx/assets/77229558/3824e96d-fffc-494e-8ce1-193e6a77c8b6

https://github.com/instant-high/deoldify-onnx/assets/77229558/543e1dd1-27da-4c63-95a9-9c0696adea51


original deoldify:

https://github.com/jantic/DeOldify

