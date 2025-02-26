import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

def generate_mlp_onnx_and_data(input_data, hidden_sizes, output_size, onnx_file, input_file, output_file):
    np.savez(input_file, input=input_data)
    
    input_size = input_data.shape[1]
    batch_size = input_data.shape[0]
    
    # 创建输入和输出张量信息
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_size])
    
    # 定义权重和偏置
    layers = []
    prev_size = input_size
    layer_inputs = ['input']
    
    for i, hidden_size in enumerate(hidden_sizes):
        weight_name = f'W{i+1}'
        bias_name = f'b{i+1}'
        output_name = f'hidden{i+1}'
        
        weight = np.random.randn(prev_size, hidden_size).astype(np.float32)
        bias = np.random.randn(hidden_size).astype(np.float32)
        
        layers.append(helper.make_tensor(weight_name, TensorProto.FLOAT, weight.shape, weight.flatten()))
        layers.append(helper.make_tensor(bias_name, TensorProto.FLOAT, bias.shape, bias.flatten()))
        
        linear_node = helper.make_node(
            'Gemm',
            inputs=[layer_inputs[-1], weight_name, bias_name],
            outputs=[output_name],
            name=f'linear{i+1}',
            alpha=1.0, beta=1.0, transB=1
        )
        
        relu_output = f'relu{i+1}'
        relu_node = helper.make_node(
            'Relu',
            inputs=[output_name],
            outputs=[relu_output],
            name=f'relu{i+1}'
        )
        
        layer_inputs.append(relu_output)
        prev_size = hidden_size
        layers.extend([linear_node, relu_node])
    
    # 输出层
    final_weight_name = 'W_out'
    final_bias_name = 'b_out'
    
    final_weight = np.random.randn(prev_size, output_size).astype(np.float32)
    final_bias = np.random.randn(output_size).astype(np.float32)
    
    layers.append(helper.make_tensor(final_weight_name, TensorProto.FLOAT, final_weight.shape, final_weight.flatten()))
    layers.append(helper.make_tensor(final_bias_name, TensorProto.FLOAT, final_bias.shape, final_bias.flatten()))
    
    output_node = helper.make_node(
        'Gemm',
        inputs=[layer_inputs[-1], final_weight_name, final_bias_name],
        outputs=['output'],
        name='output_layer',
        alpha=1.0, beta=1.0, transB=1
    )
    layers.append(output_node)
    
    # 创建计算图
    graph_def = helper.make_graph(layers, 'mlp_graph', [input_tensor], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name='mlp_model', opset_imports=[helper.make_opsetid("", 21)])
    
    # 保存 ONNX 模型
    onnx.save(model_def, onnx_file)
    print(f"ONNX model saved to {onnx_file}")
    
    # 运行 ONNX 推理
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {'input': input_data})[0]
    
    # 保存输出数据
    np.savez(output_file, output=output_data)
    print(f"Output data saved to {output_file}")

if __name__ == "__main__":
    input_data = np.random.randn(1, 128).astype(np.float32)  
    hidden_sizes = [64, 32]  
    output_size = 10  
    
    generate_mlp_onnx_and_data(input_data, hidden_sizes, output_size, 'network.onnx', 'inputs.npz', 'outputs.npz')