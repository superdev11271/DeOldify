import cv2
import numpy as np
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# inference resolution is render_factor * RENDER_BASE
RENDER_BASE = 16


class DEOLDIFY:
    def __init__(self, model_path="deoldify.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]
        # detect fp16 vs fp32 from the model's input type
        self.dtype = np.float16 if self.session.get_inputs()[0].type == "tensor(float16)" else np.float32

        
    def colorize(self, image, r_factor):
    
        # preprocess image:
        targetL = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        targetL,_,_=cv2.split(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w, channels = image.shape

        resolution = r_factor * RENDER_BASE
        image = cv2.resize(image,(resolution, resolution))
        image = image.astype(self.dtype)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(self.dtype)

        # run deoldify:
        colorized = self.session.run(None, {(self.session.get_inputs()[0].name):image})[0][0]

        # postprocess image:
        colorized = colorized.transpose(1,2,0).astype(np.float32)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        colorized = cv2.resize(colorized,(w,h))
        colorized = cv2.GaussianBlur(colorized,(13,13),0)
        colorizedLAB = cv2.cvtColor(colorized,cv2.COLOR_BGR2LAB)
        L,A,B=cv2.split(colorizedLAB)
        colorizedLAB = cv2.resize(colorizedLAB,(w, h))
        colorized = cv2.merge((targetL,A,B))
        colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
                      
        return colorized
