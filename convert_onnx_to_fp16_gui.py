import tkinter as tk
from tkinter import filedialog
import os
import onnx
from onnxconverter_common import float16

def select_model_file():
    file_path = filedialog.askopenfilename(filetypes=[("ONNX files", "*.onnx")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
        result_label.config(text="Model loaded")

def fix_cast_nodes(model):
    # the converter retypes tensors to float16 but leaves Cast 'to' attributes
    # at float32, which makes onnxruntime reject the model on load
    graph = model.graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    for v in list(graph.input) + list(graph.output):
        types[v.name] = v.type.tensor_type.elem_type
    for node in graph.node:
        if node.op_type != 'Cast':
            continue
        expected = types.get(node.output[0])
        for attr in node.attribute:
            if attr.name == 'to' and expected is not None and attr.i != expected:
                attr.i = expected


def convert_to_float16():
    model_path = entry.get()
    if model_path:
        model = onnx.load(model_path)
        result_file= os.path.splitext(model_path)[0]
        result_extension = os.path.splitext(model_path)[1]
        model_converted = result_file + "_fp16" + result_extension
        model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False,disable_shape_infer=False, op_block_list=None, node_block_list=None)
        fix_cast_nodes(model_fp16)
        onnx.save(model_fp16, model_converted)
        result_label.config(text="Model converted successfully!")

# Create the main window
root = tk.Tk()
root.title("ONNX Model Converter")

# Create widgets
label = tk.Label(root, text="Select ONNX model file:")
label.grid(row=0, column=0, padx=10, pady=5)

entry = tk.Entry(root, width=50)
entry.grid(row=0, column=1, padx=10, pady=5, columnspan=2)

browse_button = tk.Button(root, text="Browse", command=select_model_file)
browse_button.grid(row=0, column=3, padx=5, pady=5)

convert_button = tk.Button(root, text="Convert to float16", command=convert_to_float16)
convert_button.grid(row=1, column=1, columnspan=2, padx=10, pady=5)

result_label = tk.Label(root, text="")
result_label.grid(row=2, column=1, columnspan=2, padx=10, pady=5)

# Start the GUI
root.mainloop()
