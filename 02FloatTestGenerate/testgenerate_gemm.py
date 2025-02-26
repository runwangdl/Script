import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def generate_gemm_onnx_and_data(input_a, input_b, input_c, alpha, beta, onnx_file, input_file, output_file):
    np.savez(input_file, input_a=input_a, input_b=input_b, input_c=input_c)

    input_shape_a = input_a.shape  # (batch, M, K)
    input_shape_b = input_b.shape  # (batch, K, N)
    output_shape = (input_shape_a[0], input_shape_a[1], input_shape_b[2])  # (batch, M, N)

    input_tensor_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, input_shape_a)
    input_tensor_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, input_shape_b)
    input_tensor_c = helper.make_tensor_value_info('input_c', TensorProto.FLOAT, output_shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    gemm_node = helper.make_node(
        'Gemm',
        inputs=['input_a', 'input_b', 'input_c'],
        outputs=['output'],
        name='gemm_node',
        alpha=alpha,
        beta=beta,
    )

    graph_def = helper.make_graph(
        [gemm_node], 
        'gemm_graph', 
        [input_tensor_a, input_tensor_b, input_tensor_c], 
        [output_tensor]
    )

    model_def = helper.make_model(graph_def, producer_name='gemm_model', opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input_a': input_a, 'input_b': input_b, 'input_c': input_c})[0]

    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    batch, M, K, N = 1, 64, 32, 96
    alpha = 1.0
    beta = 1.0

    test_values_a = np.random.randn(batch, M, K).astype(np.float32)
    test_values_b = np.random.randn(batch, K, N).astype(np.float32)
    test_values_c = np.random.randn(batch, M, N).astype(np.float32)  

    generate_gemm_onnx_and_data(test_values_a, test_values_b, test_values_c, alpha, beta, 'network.onnx', 'inputs.npz', 'outputs.npz')
