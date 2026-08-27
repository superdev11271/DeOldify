import os
import glob
import time
import shutil
import tempfile
import subprocess

import cv2
import gradio as gr

from deoldify import DEOLDIFY

DEFAULT_MODEL = 'models/ColorizeArtistic_dyn_fp16.onnx'

_loaded = {}


def list_models():
    models = sorted(p.replace('\\', '/') for p in glob.glob('models/*.onnx'))
    return models


def get_colorizer(model, device):
    key = (model, device)
    if key not in _loaded:
        _loaded[key] = DEOLDIFY(model_path=model, device=device)
    return _loaded[key]


def to_h264(src):
    # opencv writes mp4v, which most browsers refuse to play
    if shutil.which('ffmpeg') is None:
        return src
    dst = src.replace('.mp4', '_h264.mp4')
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src,
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', dst]
    if subprocess.call(command) != 0:
        return src
    return dst


def colorize_image(image, model, device, render_factor):
    if image is None:
        raise gr.Error("Load an image first")

    colorizer = get_colorizer(model, device)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    start = time.perf_counter()
    colorized = colorizer.colorize(image, render_factor * 16)
    elapsed = time.perf_counter() - start

    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), "%.2f s" % elapsed


def colorize_video(source, model, device, render_factor, progress=gr.Progress()):
    if source is None:
        raise gr.Error("Load a video first")

    colorizer = get_colorizer(model, device)

    video = cv2.VideoCapture(source)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    result_path = os.path.join(tempfile.mkdtemp(), 'colorized.mp4')
    writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

    done = 0
    start = time.perf_counter()
    for _ in progress.tqdm(range(n_frames), desc="Colorizing"):
        ret, frame = video.read()
        if not ret:
            break
        writer.write(colorizer.colorize(frame, render_factor * 16))
        done = done + 1
    elapsed = time.perf_counter() - start

    video.release()
    writer.release()

    return to_h264(result_path), "%.2f s - %d frames - %.2f s/frame" % (elapsed, done, elapsed / max(done, 1))


with gr.Blocks(title="DeOldify ONNX") as demo:
    gr.Markdown("# DeOldify ONNX\nColorize black and white images and video.")

    with gr.Row():
        model = gr.Dropdown(list_models(), value=DEFAULT_MODEL, label="Model")
        device = gr.Radio(['cuda', 'cpu'], value='cuda', label="Device")
        # inference resolution is render_factor * 16, models need it divisible by 32
        render_factor = gr.Slider(4, 64, value=32, step=2, label="Render factor")

    with gr.Tab("Image"):
        with gr.Row():
            image_in = gr.Image(label="Source", type="numpy")
            image_out = gr.Image(label="Colorized")
        image_button = gr.Button("Colorize", variant="primary")
        image_time = gr.Textbox(label="Inference time", interactive=False)
        image_button.click(colorize_image, [image_in, model, device, render_factor], [image_out, image_time])

    with gr.Tab("Video"):
        with gr.Row():
            video_in = gr.Video(label="Source")
            video_out = gr.Video(label="Colorized")
        video_button = gr.Button("Colorize", variant="primary")
        video_time = gr.Textbox(label="Inference time", interactive=False)
        video_button.click(colorize_video, [video_in, model, device, render_factor], [video_out, video_time])


if __name__ == '__main__':
    demo.launch()
