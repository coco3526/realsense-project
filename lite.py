from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

#load the model that was trained to detect items.
interpreter  = Interpreter("detect.tflite")
#allocate the tensor
interpreter.allocate_tensors()

#retrive the input and output details contains info about the the model 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print (input_details)
# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)