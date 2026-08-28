'''
run this script in the original deoldify repo
thanks to henry ruhs - face fusion for helping
'''

import os
import argparse
import collections, collections.abc
# fastai 1.0.60 predates Python 3.10, which removed the collections ABC aliases
for _n in ('Sized', 'Iterable', 'Mapping', 'Sequence', 'Callable', 'Generator'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import torch
from deoldify.generators import gen_inference_deep
from deoldify.generators import gen_inference_wide
import torch.nn as nn
from pathlib import Path

from fastai.vision.data import normalize_funcs, imagenet_stats

# torch >=2.6 defaults torch.load to weights_only=True; the fastai checkpoint pickles a slice
torch.serialization.add_safe_globals([slice])

norm, denorm = normalize_funcs(*imagenet_stats)

class ImageScaleInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = (x.div(255.0)).type(torch.float32)
        out, _ = norm((out, out))
        return out

class ImageScaleOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = denorm(x)
        out = out.float().clamp(min=0, max=1)
        out = (out.mul(255.0)).type(torch.float32)
        return out

def export(weights_name, artistic=True, dynamic_batch=False):
    gen_inference = gen_inference_deep if artistic else gen_inference_wide
    raw_model = gen_inference(root_folder=Path('.'), weights_name=weights_name).model
    onnx_path = str(Path('./models') / (weights_name.replace('_gen', '') + '_dyn.onnx'))

    dummy_input = torch.randn(1, 3, 256, 256)

    # Wenn CUDA verfuegbar ist, auf CUDA umschalten
    dummy_input = dummy_input.to('cuda')

    final_pytorch_model = nn.Sequential(ImageScaleInput(), raw_model, ImageScaleOutput())

    axes = {2: 'height', 3: 'width'}
    if dynamic_batch:
        axes[0] = 'batch'

    torch.onnx.export(
        final_pytorch_model,
        dummy_input,
        onnx_path,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        dynamic_axes={'input': dict(axes), 'output': dict(axes)}
    )
    return onnx_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a DeOldify generator checkpoint to ONNX.')
    parser.add_argument('weights_name', nargs='?', default='ColorizeArtistic_gen',
                        help="checkpoint name in ./models, without the .pth suffix "
                             "(e.g. ColorizeArtistic_gen, ColorizeStable_gen, DeoldifyVideo_gen)")
    parser.add_argument('--artistic', action=argparse.BooleanOptionalAction, default=True,
                        help='use the deep generator (artistic); --no-artistic uses the wide one '
                             '(stable and video models)')
    parser.add_argument('--dynamic', action='store_true',
                        help='also make the batch dimension dynamic (height and width always are)')
    args = parser.parse_args()
    print(export(args.weights_name, artistic=args.artistic, dynamic_batch=args.dynamic))
