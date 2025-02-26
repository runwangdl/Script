import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def generate_gelu_onnx_and_data(input_x, onnx_file, input_file, output_file):
    np.savez(input_file, input_x=input_x)

    input_shape = input_x.shape
    output_shape = input_shape

    input_tensor_x = helper.make_tensor_value_info('input_x', TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    gelu_node = helper.make_node(
        'Gelu',
        inputs=['input_x'],
        outputs=['output'],
        name='gelu_node'
    )

    graph_def = helper.make_graph(
        [gelu_node], 
        'gelu_graph', 
        [input_tensor_x], 
        [output_tensor]
    )

    model_def = helper.make_model(graph_def, producer_name='gelu_model', opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input_x': input_x})[0]

    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    test_values_x = np.random.randn(1, 64, 32).astype(np.float32)  

    generate_gelu_onnx_and_data(test_values_x, 'network.onnx', 'inputs.npz', 'outputs.npz')
