#from keras.backend import clear_session
import numpy as np
import tensorflow as tf
#clear_session()
np.set_printoptions(suppress=True)
input_graph_name = '/home/pan/master-thesis-in-mrt/test_raspberry/train_result/network/large_model_final.h5'
output_graph_name = '/home/pan/master-thesis-in-mrt/test_raspberry/train_result/network/MobileNetV3_large_weights_89.50%_89.99%_93.68%.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file=input_graph_name)
converter.post_training_quantize = False   #是否量化的选项
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)
print("generate:",output_graph_name)