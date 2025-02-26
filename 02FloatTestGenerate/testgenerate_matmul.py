import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def generate_matmul_onnx_and_data(input_a, input_b, onnx_file, input_file, output_file):
    
    np.savez(input_file, input_a=input_a, input_b=input_b)

    input_shape_a = input_a.shape
    input_shape_b = input_b.shape
    output_shape = (input_shape_a[0], input_shape_a[1], input_shape_b[2])

    input_tensor_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, input_shape_a)
    input_tensor_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, input_shape_b)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input_a', 'input_b'],
        outputs=['output'],
        name='matmul_node'
    )

    graph_def = helper.make_graph(
        [matmul_node], 
        'matmul_graph', 
        [input_tensor_a, input_tensor_b], 
        [output_tensor]
    )

    model_def = helper.make_model(graph_def, producer_name='matmul_model', opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input_a': input_a, 'input_b': input_b})[0]

    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    test_values_a = np.random.randn(1, 64, 32).astype(np.float32)  
    test_values_b = np.random.randn(1, 32, 96).astype(np.float32)  
    
    generate_matmul_onnx_and_data(test_values_a, test_values_b, 'network.onnx', 'inputs.npz', 'outputs.npz')
