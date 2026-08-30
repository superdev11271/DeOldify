import os
import cv2
from deoldify_onnx import DEOLDIFY
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", required=True, help="path to source image")
    parser.add_argument("--result", help="path to result image, default: source name + _colorized")
    parser.add_argument("--model", default='models/ColorizeArtistic_dyn_fp16.onnx', help="path to onnx model")
    parser.add_argument("--device", default='cuda', help="cuda or cpu")
    parser.add_argument("--render_factor", type=int, default=35, help=" - ")
    opt = parser.parse_args()

    colorizer = DEOLDIFY(model_path=opt.model, device=opt.device)

    image = cv2.imread(opt.source)

    colorized = colorizer.colorize(image, opt.render_factor)

    result = opt.result
    if result is None:
        name, ext = os.path.splitext(opt.source)
        result = name + "_colorized" + ext
    cv2.imwrite(result, colorized)

    cv2.imshow("Colorized image saved - press any key", colorized)
    cv2.waitKey()


if __name__ == '__main__':
    main()
