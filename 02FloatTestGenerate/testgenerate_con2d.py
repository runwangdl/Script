import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def generate_conv2d_onnx_and_data(input_data, kernel_size, stride, padding, 
                                  out_channels, use_bias, group, dilation,
                                  onnx_file, input_file, output_file):

    np.savez(input_file, input=input_data)

    batch_size, in_channels, height, width = input_data.shape

    effective_kernel = (kernel_size - 1) * dilation + 1
    output_height = (height + 2 * padding - effective_kernel) // stride + 1
    output_width = (width + 2 * padding - effective_kernel) // stride + 1
    output_shape = (batch_size, out_channels, output_height, output_width)

    weight_shape = (out_channels, in_channels // group, kernel_size, kernel_size)
    weight = np.random.randn(*weight_shape).astype(np.float32)

    bias = np.random.randn(out_channels).astype(np.float32) if use_bias else None


    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_data.shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, weight.shape, weight.flatten())

    initializers = [weight_tensor]

 
    if use_bias:
        bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, bias.shape, bias.flatten())
        initializers.append(bias_tensor)
        conv_inputs = ['input', 'weight', 'bias']
    else:
        conv_inputs = ['input', 'weight']


    conv_node = helper.make_node(
        'Conv',
        inputs=conv_inputs,
        outputs=['output'],
        name='conv_node',
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[padding, padding, padding, padding],
        group=group,
        dilations=[dilation, dilation]  
    )
 
    graph_def = helper.make_graph(
        [conv_node], 
        'conv_graph', 
        [input_tensor], 
        [output_tensor],
        initializer=initializers
    )

  
    model_def = helper.make_model(graph_def, producer_name='conv_model', opset_imports=[helper.make_opsetid("", 21)])

    
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")

   
    ort_session = ort.InferenceSession(onnx_file)
    ort_inputs = {'input': input_data}
    output_data = ort_session.run(None, ort_inputs)[0]

    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    test_values = np.random.randn(1, 3, 16, 16).astype(np.float32)  
    
    kernel_size = 3  
    stride = 1 
    padding = 1  
    out_channels = 64  
    use_bias = False 
    group = 1 
    dilation = 1  

    generate_conv2d_onnx_and_data(test_values, kernel_size, stride, padding, 
                                  out_channels, use_bias, group, dilation,
                                  'network.onnx', 'inputs.npz', 'outputs.npz')
