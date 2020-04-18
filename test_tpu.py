import argparse
import time
import cv2
from PIL import Image

import util_tpu
import tflite_runtime.interpreter as tflite
import platform
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 3))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 3))
    return time_str

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', default='/home/pandongwei/master-thesis-in-mrt/test_raspberry/train_result/output_tflite_graph_edgetpu.tflite',
      help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--image_path', default='/home/pi/usbdisk/mrt-storage/data/dataset-test',
      help='Image to be classified.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  args = parser.parse_args()

  DATA_CLASS_NAMES = {
      "bicycle-lane": 0,
      "bicycle-lane-and-pedestrian": 1,
      "car-lane": 2,
      "pedestrian": 3
  }

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  output_details = interpreter.get_output_details()
  size = util_tpu.input_size(interpreter)
  # Test model
  start_1 = cv2.getTickCount()

  correct = 0
  sum = 0.0
  #images_path = args.image_path
  images_path = '/home/pandongwei/master-thesis-in-mrt/dataset-test'
  for root, dirs, files in os.walk(images_path, topdown=True):
      for name in dirs:
          path_front = os.path.join(root, name)
          class_name = name
          a = os.listdir(path_front)
          a.sort()
          for f in a:
              start = cv2.getTickCount()
              image_path = os.path.join(path_front, f)
              image = Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS)
              util_tpu.set_input(interpreter, image)
              interpreter.invoke()
              output_data = interpreter.get_tensor(output_details[0]['index'])
              output = output_data.argmax()
              #output = util_tpu.get_output(interpreter, args.top_k, args.threshold)
              label = DATA_CLASS_NAMES.get(class_name, 0)
              if output == label:
                  correct += 1
              sum += 1
              time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
              print((sum, '   ', output, "   Took {}".format(format_time(time))))
  acc = float(correct / sum)
  print(correct)
  print('the accuracy is: {}%'.format(round(acc * 100, 2)))

  time_1 = (cv2.getTickCount() - start_1) / cv2.getTickFrequency()
  print("Took totally {}".format(format_time(time_1)))

  # size = util_tpu.input_size(interpreter)
  # image = Image.open(args.image_path).convert('RGB').resize(size, Image.ANTIALIAS)
  # util_tpu.set_input(interpreter, image)
  #
  # print('----INFERENCE TIME----')
  # print('Note: The first inference on Edge TPU is slow because it includes',
  #       'loading the model into Edge TPU memory.')
  # for _ in range(args.count):
  #   start = time.perf_counter()
  #   interpreter.invoke()
  #   inference_time = time.perf_counter() - start
  #   classes = util_tpu.get_output(interpreter, args.top_k, args.threshold)
  #   print('%.1fms' % (inference_time * 1000))
  #
  # print('-------RESULTS--------')
  # for klass in classes:
  #   print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))


if __name__ == '__main__':
  main()
