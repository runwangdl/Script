import re
import numpy as np
import onnx
import onnxruntime as ort
import torch
from collections import defaultdict
from onnx import helper, numpy_helper
from src import *

def make_c_name(op_type, count=0):
    return f"node_{op_type}_{count}"

def get_other_input_shape(model, node, bias_name_pattern):
    """获取Add节点非bias输入的shape"""
    for vi in model.graph.value_info:
        if vi.name in node.input and not re.search(bias_name_pattern, vi.name):
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None

def modify_add_nodes(model_path):
    model = onnx.load(model_path)
    nodes_info = [
        {
            'node_pattern': r'node_.*_linear1_Add',
            'old_bias_pattern': r'classifier_blocks_0_linear1_bias',
            'new_bias_name': 'classifier_blocks_0_linear1_bias',  
            'create_new': False
        },
        {
            'node_pattern': r'node_.*_linear2_Add',
            'old_bias_pattern': r'classifier_blocks_0_pre_norm_bias',
            'new_bias_name': 'node_classifier_blocks_0_linear2_Add_bias',
            'create_new': True
        },
        {
            'node_pattern': r'node_.*_self_attn_proj_Add',
            'old_bias_pattern': r'classifier_blocks_0_pre_norm_bias',
            'new_bias_name': 'node_classifier_blocks_0_self_attn_proj_Add_bias',
            'create_new': True
        }
    ]
    
    for info in nodes_info:
        print(f"\nProcessing nodes matching: {info['node_pattern']}")
        target_node = None
        for node in model.graph.node:
            if node.op_type == 'Add' and re.search(info['node_pattern'], node.output[0]):
                target_node = node
                break
        
        if not target_node:
            print(f"Cannot find matching node for pattern {info['node_pattern']}")
            continue
        
        print(f"Current inputs: {target_node.input}")
        
        other_input_shape = get_other_input_shape(model, target_node, info['old_bias_pattern'])
        if not other_input_shape:
            print("Cannot find shape for non-bias input")
            continue
        
        print(f"Target shape: {other_input_shape}")
        
        old_bias = None
        for init in model.graph.initializer:
            if re.search(info['old_bias_pattern'], init.name):
                old_bias = init
                break
                
        if not old_bias:
            print(f"Cannot find bias matching pattern {info['old_bias_pattern']}")
            continue
        
        bias_data = numpy_helper.to_array(old_bias)
        new_bias_data = np.broadcast_to(bias_data, other_input_shape)
        new_bias = numpy_helper.from_array(new_bias_data, info['new_bias_name'])
        
        if info['create_new']:
            model.graph.initializer.append(new_bias)
            if target_node.input[0] == old_bias.name:
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
        if node.op_type == 'Gemm' and re.search(r'node_.*_fc_Gemm', node.output[0]):
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
                if re.search(r'classifier_fc_bias', init.name):
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
