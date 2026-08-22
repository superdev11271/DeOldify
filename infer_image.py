import cv2
from deoldify import DEOLDIFY
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--source_image", default='source.jpg', help="path to source image")
parser.add_argument("--render_factor", type=int, default=16, help=" - ")
opt = parser.parse_args()


render_factor = opt.render_factor * 16

colorizer = DEOLDIFY(model_path="models/ColorizeArtistic_dyn_fp16.onnx", device="cuda")

image = cv2.imread(opt.source_image)

colorized = colorizer.colorize(image, render_factor)

# cv2.imwrite(opt.result_image, colorized) 
cv2.imshow("Colorized image saved - press any key",colorized)
cv2.waitKey() 
