import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
from transformers import AutoTokenizer, AutoProcessor
from modeling_idefics3_export import Idefics3ForConditionalGenerationExport
from qwen_vl_utils import process_vision_info
import numpy as np
import os
import sys


def export_onnx(model, input, input_names, output_names, onnx_output):

    torch.onnx.export(
        model,
        input,
        onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )

    onnx_model = onnx.load(onnx_output)
    print("IR 版本:", onnx_model.ir_version)
    print("操作集:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_output)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))


checkpoint_dir = sys.argv[1] if len(sys.argv) >= 2 else "../SmolVLM-256M-Instruct"
# default: Load the model on the available device(s)
model = Idefics3ForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cpu"
)

export_model = model.model.vision_model
export_model.forward = export_model.forward_export
device = torch.device("cpu")


input = torch.ones((1, 3, 512, 512), dtype=torch.float32).to(device)

input_names = ["pixel_values"]

onnx_output = f"SmolVLM-256M-Instruct_vision.onnx"

output_names = [f"last_hidden_state"]


export_onnx(export_model, input, input_names, output_names, onnx_output)
