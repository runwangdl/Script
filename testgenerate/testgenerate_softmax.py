import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def generate_softmax_onnx_and_data(input_data, axis, onnx_file, input_file, output_file):
   
    np.savez(input_file, input=input_data)

  
    input_shape = input_data.shape

    
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)  


    softmax_node = helper.make_node(
        'Softmax',
        inputs=['input'],
        outputs=['output'],
        name='softmax_node',
        axis=axis  
    )

    graph_def = helper.make_graph(
        [softmax_node], 
        'softmax_graph', 
        [input_tensor], 
        [output_tensor]
    )


    model_def = helper.make_model(graph_def, producer_name='softmax_model', opset_imports=[helper.make_opsetid("", 13)])


    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

   
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input': input_data})[0]


    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    test_values = np.random.randn(1, 1, 64, 64).astype(np.float32)  
    
    axis = 3

    generate_softmax_onnx_and_data(test_values, axis, 'network.onnx', 'inputs.npz', 'outputs.npz')
