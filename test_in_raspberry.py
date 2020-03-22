import numpy as np
import tensorflow as tf
import os
import cv2
from utils import *


def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 3))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 3))
    return time_str

def main():
    start_1 = cv2.getTickCount()
    DATA_CLASS_NAMES = {
        "bicycle-lane": 0,
        "bicycle-lane-and-pedestrian": 1,
        "car-lane": 2,
        "pedestrian": 3
    }

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="logs/fit/20191223-124218/weights_best_pre_93.67%_94.32%_96.77%_3.7.tflite")
    #interpreter = tf.lite.Interpreter(model_path="logs/fit/20191223-124218/mobilenet_v2_1.0_224.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model
    correct = 0
    sum = 0.0
    #input_shape = input_details[0]['shape']
    test_dir = "/home/pan/master-thesis-in-mrt/data/dataset-evaluation"
    #test_dir = '/media/pandongwei/Extreme SSD/mrt-storage/data/dataset-test'
    #test_dir = '/home/pi/usbdisk/mrt-storage/data/dataset-test'

    print('loading image.....')
    start = cv2.getTickCount()
    path, class_names = generate_path(test_dir)
    test_imgs_data = load_images(path, class_names)
    print_duration(start)
    class_labels = np.int32([img_data.class_number for img_data in test_imgs_data])
    # samples = stack_array([[cv2.cvtColor(img_data.img,cv2.COLOR_BGR2RGB)] for img_data in test_imgs_data])
    samples = []
    for img_data in test_imgs_data:
        samples.append(cv2.cvtColor(img_data.img, cv2.COLOR_BGR2RGB))
    samples = np.asanyarray(samples)
    print_duration(start)
    print('finish loading image')
    samples = samples / 255.

    for i in range(len(samples)):
        start = cv2.getTickCount()
        input_shape = (1,224,224,3)
        #input_data = np.reshape(img, input_shape)
        input_data = np.float32(np.reshape(samples[i] / 255.,input_shape))
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output = output_data.argmax()
        label = class_labels[i]
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