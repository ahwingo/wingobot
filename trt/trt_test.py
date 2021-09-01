#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

saved_model_dir = "testing_pb_export_savedmodel"
trt_model_dir = "testing_pb_export_savedmodel_trt"

fake_input = tf.constant(np.random.random((64, 19, 13, 13)), dtype=float)
num_trials = 168


def input_generator():
    yield [fake_input]


# converter = trt.TrtGraphConverter(input_saved_model_dir=saved_model_dir,
#                                   input_saved_model_tags = ["value/Tanh", "policy/Softmax"])
start = time.time()
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 33))
conversion_params = conversion_params._replace(precision_mode="FP32")
#conversion_params = conversion_params._replace(use_calibration=True)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=conversion_params)
#converter.convert(calibration_input_fn=input_generator)
converter.convert()
#converter.build(input_generator)
converter.save(trt_model_dir)


print("\n\n\nTesting inference speed of the TRT optimized model.")
saved_model_loaded = tf.saved_model.load(trt_model_dir, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']
output = infer(fake_input)
end = time.time()
print("Time to convert to TRT model an call the first inference was {} seconds.".format(end - start))
output = infer(fake_input)
print(output)

trt_avg = 0.0
for count in range(num_trials):
    start = time.time()
    output = infer(fake_input)
    end = time.time()
    trt_time = end - start
    trt_avg = (trt_avg * count + trt_time) / (count + 1)
print("Average inference with the TRT optimized model took {} seconds.".format(trt_avg))

print("\n\n\nTesting inference speed of the original input model.")
saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']
output = infer(fake_input)
output = infer(fake_input)
non_trt_avg = 0.0
for count in range(num_trials):
    start = time.time()
    output = infer(fake_input)
    end = time.time()
    non_trt_time = end - start
    non_trt_avg = (non_trt_avg * count + non_trt_time) / (count + 1)

print("Average inference with the non TRT optimized model took {} seconds.".format(non_trt_avg))

print("\n\n\nThe total speedup is {}.".format(non_trt_avg / trt_avg))
