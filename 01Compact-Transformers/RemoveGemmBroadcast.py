import onnx
import numpy as np
from onnx import numpy_helper

def modify_onnx_bias(input_onnx_path, output_onnx_path):
    # Load the ONNX model
    model = onnx.load(input_onnx_path)
    graph = model.graph
    
    # Iterate over initializers to find classifier_fc_bias
    for initializer in graph.initializer:
        if initializer.name == "classifier_fc_bias":
            # Convert to numpy array
            bias_array = numpy_helper.to_array(initializer)
            
            # Check original shape
            if bias_array.shape == (10,):
                # Reshape to (1, 10) to explicitly keep 1
                new_bias_array = bias_array.reshape((1, 10))
                
                # Convert back to ONNX tensor
                new_initializer = numpy_helper.from_array(new_bias_array, name="classifier_fc_bias")
                
                # Replace the initializer in the graph
                graph.initializer.remove(initializer)
                graph.initializer.append(new_initializer)
                print("Modified classifier_fc_bias shape from (10,) to (1,10)")
                break
    else:
        print("classifier_fc_bias not found in the model.")
        return
    
    # Save the modified ONNX model
    onnx.save(model, output_onnx_path)
    print(f"Modified model saved to {output_onnx_path}")

# Example usage
modify_onnx_bias("/app/Deeploy/DeeployTest/Tests/CCT/network.onnx", "/app/Deeploy/DeeployTest/Tests/CCT/network.onnx")
