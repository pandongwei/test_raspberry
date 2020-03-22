import numpy as np
import tensorflow as tf
import os
import cv2
from threading import Thread
from queue import Queue
from multiprocessing import Pool

def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 3))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 3))
    return time_str

def main():
    DATA_CLASS_NAMES = {
        "bicycle-lane": 0,
        "bicycle-lane-and-pedestrian": 1,
        "car-lane": 2,
        "pedestrian": 3
    }
    image_path = '/home/pi/usbdisk/mrt-storage/data/dataset-test'
    model_path = 'logs/fit/20191223-124218/weights_best_pre_93.67%_94.32%_96.77%_3.7.tflite'

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    #interpreter = tf.lite.Interpreter(model_path="logs/fit/20191223-124218/mobilenet_v2_1.0_224.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model
    start_1 = cv2.getTickCount()

    correct = 0
    sum = 0.0
    for root, dirs, files in os.walk(image_path, topdown=True):
        for name in dirs:
            path_front = os.path.join(root, name)
            class_name = name
            a = os.listdir(path_front)
            a.sort()
            for f in a:
                start = cv2.getTickCount()
                image_path = os.path.join(path_front, f)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_shape = (1,224,224,3)
                #input_data = np.reshape(img, input_shape)
                input_data = np.float32(np.reshape(img / 255.,input_shape))
                #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                output = output_data.argmax()
                label = DATA_CLASS_NAMES.get(class_name,0)
                if output == label:
                    correct += 1
                sum += 1
                time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
                print((sum,'   ',output,"   Took {}".format(format_time(time))))
    acc = float(correct/sum)
    print(correct)
    print('the accuracy is: {}%'.format(round(acc*100,2)))

    time_1 = (cv2.getTickCount() - start_1) / cv2.getTickFrequency()
    print("Took totally {}".format(format_time(time_1)))

if __name__ == '__main__':
    main()
