"""DeOldify ONNX colorizer: preprocess -> session -> postprocess."""
import cv2
import numpy as np
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# inference resolution is render_factor * RENDER_BASE
RENDER_BASE = 16


class DEOLDIFY():
    """Colorize grayscale images with an exported DeOldify ONNX model.

    Args:
        model_path (str): Path to the .onnx model. fp32 and fp16 models are
            both supported; the input dtype is read from the model.
        device (str): 'cuda' or 'cpu' (default). 'cuda' falls back to CPU when
            no CUDA execution provider is available.
    """

    def __init__(self, model_path='deoldify.onnx', device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.resolution = inp.shape[-2:]
        # fp16 models expect a float16 tensor; everything else stays float32
        self.dtype = np.float16 if inp.type == 'tensor(float16)' else np.float32
        # the batch axis is symbolic only when exported with --dynamic, otherwise it is pinned to 1
        self.dynamic_batch = not isinstance(inp.shape[0], int)

    def colorize(self, image, r_factor):
        """Colorize one BGR uint8 image and return a BGR uint8 image."""
        return self.colorize_batch([image], r_factor)[0]

    def colorize_batch(self, images, r_factor):
        """Colorize a list of BGR uint8 images.

        A batch larger than 1 needs a model exported with `--dynamic`.
        """
        resolution = r_factor * RENDER_BASE
        batch = np.stack([self._preprocess(image, resolution) for image in images])

        if self.dynamic_batch:
            colorized = self.session.run(None, {self.input_name: batch})[0]
        else:
            # model has a fixed batch of 1, so feed the images one at a time
            colorized = np.concatenate(
                [self.session.run(None, {self.input_name: batch[i:i + 1]})[0] for i in range(len(batch))])

        return [self._postprocess(c, image) for c, image in zip(colorized, images)]

    def _preprocess(self, image, resolution):
        """BGR uint8 HWC -> grayscale-as-RGB CHW in the model's dtype."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (resolution, resolution))
        return image.transpose((2, 0, 1)).astype(self.dtype)

    def _postprocess(self, colorized, source):
        """Take the chroma from the model output and the luma from the source image."""
        targetL, _, _ = cv2.split(source)
        h, w, channels = source.shape

        colorized = colorized.transpose(1, 2, 0).astype(np.float32)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        colorized = cv2.resize(colorized, (w, h))
        colorized = cv2.GaussianBlur(colorized, (13, 13), 0)
        colorizedLAB = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(colorizedLAB)
        colorizedLAB = cv2.resize(colorizedLAB, (w, h))
        colorized = cv2.merge((targetL, A, B))
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

        return colorized
