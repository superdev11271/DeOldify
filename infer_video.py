import os
import cv2
import subprocess
import platform

from deoldify_onnx import DEOLDIFY
from argparse import ArgumentParser
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", required=True, help="path to source video")
    parser.add_argument("--result", help="path to result video, default: source name + _colorized")
    parser.add_argument("--model", default='models/ColorizeArtistic_dyn_fp16.onnx', help="path to onnx model")
    parser.add_argument("--audio", default=False, action="store_true", help="Keep audio")
    parser.add_argument("--device", default='cuda', help="cuda or cpu")
    parser.add_argument("--render_factor", type=int, default=32, help=" - ")
    opt = parser.parse_args()

    colorizer = DEOLDIFY(model_path=opt.model, device=opt.device)

    result_path = opt.result
    if result_path is None:
        name, ext = os.path.splitext(opt.source)
        result_path = name + "_colorized" + ext

    video = cv2.VideoCapture(opt.source)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if opt.audio:
        writer = cv2.VideoWriter('temp.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
    else:
        writer = cv2.VideoWriter(result_path,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

    for frame_idx in tqdm(range(n_frames)):

        ret, frame = video.read()
        if not ret:
            break

        result = colorizer.colorize(frame, opt.render_factor)

        writer.write(result)
        cv2.imshow ("Result",result)
        k = cv2.waitKey(1)
        if k == 27:
            writer.release()
            break

    if opt.audio:
        # lossless remuxing audio/video
        command = 'ffmpeg.exe -y -vn -i ' + '"' + opt.source + '"' + ' -an -i ' + 'temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + result_path + '"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.remove('temp.mp4')


if __name__ == '__main__':
    main()
