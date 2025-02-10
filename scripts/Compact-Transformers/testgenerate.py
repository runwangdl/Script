import re

import numpy as np
import onnx
import onnxruntime as ort
import torch
from src import *


def make_c_name(name):

    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    if name[0].isdigit() or name[0] == '_':
        name = 'node' + name
    return name


# filepath: /app/Deeploy/DeeployTest/Tests/CCT/Compact-Transformers/testgenerate.py
def export_and_optimize_cct_model_to_onnx():
    try:
        model = cct_1_3x1_32(pretrained = False, img_size = 32, num_classes = 10)

        # checkpoint_url = "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth"
        # state_dict = load_state_dict_from_url(checkpoint_url, progress = True)
        # model.load_state_dict(state_dict)

        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)

        output_onnx_file = "/app/Deeploy/DeeployTest/Tests/CCT/network.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_file,
            opset_version = 12,  # ONNX opset version
            input_names = ["input"],
            output_names = ["output"],
            dynamic_axes = {
                "input": {
                    0: "batch_size"
                },
                "output": {
                    0: "batch_size"
                }
            },
        )

        print(f"ONNX model exported to {output_onnx_file}")
        onnx_model = onnx.load(output_onnx_file)

        for node in onnx_model.graph.node:
            node.name = make_c_name(node.name)
            for i, input_name in enumerate(node.input):
                node.input[i] = make_c_name(input_name)
            for i, output_name in enumerate(node.output):
                node.output[i] = make_c_name(output_name)

        for input in onnx_model.graph.input:
            input.name = make_c_name(input.name)
        for output in onnx_model.graph.output:
            output.name = make_c_name(output.name)

        for init in onnx_model.graph.initializer:
            init.name = make_c_name(init.name)

        # Save the modified ONNX model
        onnx.save(onnx_model, output_onnx_file)
        print(f"Modified ONNX model saved to {output_onnx_file}")
        # Generate random input data
        input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)

        # Save input data to npz file
        input_file = "/app/Deeploy/DeeployTest/Tests/CCT/inputs.npz"
        np.savez(input_file, input = input_data)
        print(f"Input data saved to {input_file}")

        # Generate output data using the model (requires ONNX runtime)
        ort_session = ort.InferenceSession(output_onnx_file)
        output_data = ort_session.run(None, {'input': input_data})[0]

        # Save output data to npz file
        output_file = "/app/Deeploy/DeeployTest/Tests/CCT/outputs.npz"
        np.savez(output_file, output = output_data)
        print(f"Output data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    export_and_optimize_cct_model_to_onnx()
