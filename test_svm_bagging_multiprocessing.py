from utils import *
from joblib import load
import cv2
from kalman_filter import *
from multiprocessing import Process, Queue, Pool,freeze_support,Manager
import os

def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 3))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 3))
    return time_str

def process_data(img_path, label):
    global correct
    global sum_image
    while not exitFlag:
        start = cv2.getTickCount()
        if not img_path.empty():
            path = img_path.get()
            img = cv2.imread(path)
            img_hog = cv2.resize(img,
                                 (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]),
                                 interpolation=cv2.INTER_AREA)
            hog_descriptor = cv2.HOGDescriptor().compute(img_hog)
            hog_descriptor = hog_descriptor.reshape((1, 3780))
            result = clf.predict_proba(hog_descriptor)
            output = result.argmax()
            img_label = label.get()
            if output == img_label:
                correct += 1
            sum_image += 1
            time_2 = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            print((correct, sum_image, '   ', output, "   Took {}".format(format_time(time_2))))
        else:
            break
    return [correct, sum_image]

if __name__ == '__main__':

    #测试部分开始
    DATA_CLASS_NAMES = {
        "bicycle-lane": 0,
        "bicycle-lane-and-pedestrian": 1,
        "car-lane": 2,
        "pedestrian": 3
    }
    image_path = '/home/pi/master-thesis/dataset-test'
    #image_path = '/home/pan/master-thesis-in-mrt/data/dataset-test'
    model_path_svm = 'train_result/traditional_method/train_whole_dataset_svm_3,7.joblib'
    #model_path_bagging = 'train_result/traditional_method/train_whole_dataset_bagging_3,7.joblib'

    # Test model
    correct = 0
    sum_image = 0.0
    exitFlag = 0
    workQueue = []
    labelQueue = []
    for i in range(4):
        m = Manager()
        workQueue.append(m.Queue(5000))
        labelQueue.append(m.Queue(5000))
    input_shape = (1, 224, 224, 3)

    clf = load(model_path_svm)

    start_1 = cv2.getTickCount()
    # 填充队列
    for root, dirs, files in os.walk(image_path, topdown=True):
        for i in range(len(dirs)):
            path_front = os.path.join(root, dirs[i])
            class_name = dirs[i]
            nameList = os.listdir(path_front)
            nameList.sort()
            for path in nameList:
                img_path = os.path.join(path_front, path)
                workQueue[i].put(img_path)
                label = DATA_CLASS_NAMES.get(class_name, 0)
                labelQueue[i].put(label)

    freeze_support()
    pool = Pool()
    results = []
    for i in range(0, 4):
        result = pool.apply_async(func=process_data, args=(workQueue[i], labelQueue[i]))
        results.append(result)
    pool.close()
    pool.join()
    pool.close()

    time_1 = (cv2.getTickCount() - start_1) / cv2.getTickFrequency()
    print("Took totally {}".format(format_time(time_1)))

    cor_img = 0
    sum_img = 0
    for result in results:
        cor, sum = result.get()
        cor_img += cor
        sum_img += sum
    acc = float(cor_img / sum_img)
    print(cor_img)
    print('the accuracy is: {}%'.format(round(acc * 100, 2)))

################################################################################



