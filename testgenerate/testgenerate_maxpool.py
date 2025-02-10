import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def generate_maxpool2d_onnx_and_data(input_data, kernel_size, stride, padding, ceil_mode, onnx_file, input_file, output_file):

    np.savez(input_file, input=input_data)

    batch_size, in_channels, height, width = input_data.shape

   
    if ceil_mode:
        output_height = int(np.ceil((height + 2 * padding - kernel_size) / stride + 1)) 
        output_width = int(np.ceil((width + 2 * padding - kernel_size) / stride + 1))
    else:
        output_height = int(np.floor((height + 2 * padding - kernel_size) / stride + 1))
        output_width = int(np.floor((width + 2 * padding - kernel_size) / stride + 1))

    output_shape = (batch_size, in_channels, output_height, output_width)

   
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_data.shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape) 


    maxpool_node = helper.make_node(
        'MaxPool',
        inputs=['input'],
        outputs=['output'],
        name='maxpool_node',
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[padding, padding, padding, padding],
        ceil_mode=int(ceil_mode)  
    )

   
    graph_def = helper.make_graph(
        [maxpool_node], 
        'maxpool_graph', 
        [input_tensor], 
        [output_tensor]
    )

    
    model_def = helper.make_model(graph_def, producer_name='maxpool_model', opset_imports=[helper.make_opsetid("", 21)])

   
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

    
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input': input_data})[0]

    
    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    test_values = np.random.randn(1, 1, 16, 16).astype(np.float32)  
    
    kernel_size = 3  
    stride = 2 
    padding = 1  
    ceil_mode = False 

    generate_maxpool2d_onnx_and_data(test_values, kernel_size, stride, padding, ceil_mode, 'network.onnx', 'inputs.npz', 'outputs.npz')
