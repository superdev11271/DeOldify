import cv2
from deoldify import DEOLDIFY
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", required=True, help="path to source image")
    parser.add_argument("--model", default='models/ColorizeArtistic_dyn_fp16.onnx', help="path to onnx model")
    parser.add_argument("--device", default='cuda', help="cuda or cpu")
    parser.add_argument("--render_factor", type=int, default=32, help=" - ")
    opt = parser.parse_args()

    render_factor = opt.render_factor * 16

    colorizer = DEOLDIFY(model_path=opt.model, device=opt.device)

    image = cv2.imread(opt.source)

    colorized = colorizer.colorize(image, render_factor)

    # cv2.imwrite(opt.result_image, colorized)
    cv2.imshow("Colorized image saved - press any key", colorized)
    cv2.waitKey()


if __name__ == '__main__':
    main()
