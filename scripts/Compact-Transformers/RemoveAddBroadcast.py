import onnx
import numpy as np
from onnx import helper, numpy_helper

def get_other_input_shape(model, node, bias_name):
    """获取Add节点非bias输入的shape"""
    for vi in model.graph.value_info:
        if vi.name in node.input and bias_name not in vi.name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None

def modify_add_nodes(model_path):
    # 加载模型
    model = onnx.load(model_path)
    
    # 需要处理的节点信息
    nodes_info = [
        {
            'node_pattern': 'node_classifier_blocks_0_linear1_Add',
            'old_bias_name': 'classifier_blocks_0_linear1_bias',
            'new_bias_name': 'classifier_blocks_0_linear1_bias',  # 同名，直接修改原始bias
            'create_new': False
        },
        {
            'node_pattern': 'node_classifier_blocks_0_linear2_Add',
            'old_bias_name': 'classifier_blocks_0_pre_norm_bias',
            'new_bias_name': 'node_classifier_blocks_0_linear2_Add_bias',
            'create_new': True
        },
        {
            'node_pattern': 'node_classifier_blocks_0_self_attn_proj_Add',
            'old_bias_name': 'classifier_blocks_0_pre_norm_bias',
            'new_bias_name': 'node_classifier_blocks_0_self_attn_proj_Add_bias',
            'create_new': True
        }
    ]
    
    # 处理Add节点
    for info in nodes_info:
        print(f"\nProcessing node: {info['node_pattern']}")
        
        target_node = None
        for node in model.graph.node:
            if node.op_type == 'Add' and info['node_pattern'] in node.output[0]:
                target_node = node
                break
        
        if not target_node:
            print(f"Cannot find node {info['node_pattern']}")
            continue
            
        print(f"Current inputs: {target_node.input}")
        
        other_input_shape = get_other_input_shape(model, target_node, info['old_bias_name'])
        if not other_input_shape:
            print(f"Cannot find shape for non-bias input")
            continue
            
        print(f"Target shape: {other_input_shape}")
        
        old_bias = None
        for init in model.graph.initializer:
            if init.name == info['old_bias_name']:
                old_bias = init
                break
                
        if not old_bias:
            print(f"Cannot find bias {info['old_bias_name']}")
            continue
        
        bias_data = numpy_helper.to_array(old_bias)
        new_bias_data = np.broadcast_to(bias_data, other_input_shape)
        new_bias = numpy_helper.from_array(new_bias_data, info['new_bias_name'])
        
        if info['create_new']:
            model.graph.initializer.append(new_bias)
            if target_node.input[0] == info['old_bias_name']:
                target_node.input[0] = info['new_bias_name']
            else:
                target_node.input[1] = info['new_bias_name']
        else:
            model.graph.initializer.remove(old_bias)
            model.graph.initializer.append(new_bias)
            
        print(f"Modified/Created bias: {info['new_bias_name']}")
        print(f"New shape: {new_bias_data.shape}")
        print(f"Updated node inputs: {target_node.input}")
    print("\nProcessing node: node_classifier_fc_Gemm")
    target_node = None
    for node in model.graph.node:
        if node.op_type == 'Gemm' and 'node_classifier_fc_Gemm' in node.output[0]:
            target_node = node
            break
    
    if target_node:
        print(f"Found node_classifier_fc_Gemm: {target_node.input}")
        
        output_shape = None
        for vi in model.graph.value_info:
            if vi.name == target_node.output[0]:
                output_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                break
        
        if output_shape:
            for init in model.graph.initializer:
                if init.name == 'classifier_fc_bias':
                    bias_data = numpy_helper.to_array(init)
                    new_bias_data = np.reshape(bias_data, output_shape)
                    model.graph.initializer.remove(init)
                    model.graph.initializer.append(numpy_helper.from_array(new_bias_data, 'classifier_fc_bias'))
                    print(f"Updated classifier_fc_bias shape to {output_shape}")
        else:
            print("Cannot determine output shape for node_classifier_fc_Gemm")
    else:
        print("Cannot find node_classifier_fc_Gemm")
    
  
    output_path = model_path
    onnx.save(model, output_path)
    print(f"\nSaved modified model to {output_path}")

if __name__ == "__main__":
    modify_add_nodes("/app/Deeploy/DeeployTest/Tests/CCT/network.onnx")
