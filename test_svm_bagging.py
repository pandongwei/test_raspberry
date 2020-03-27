from utils import *
from joblib import load
import cv2
from kalman_filter import *
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

def main():

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

    path, class_names = generate_path(image_path)

    test_imgs_data = load_images(path, class_names)
    #test_imgs_data = shuffle(test_imgs_data)
    print("testing with {} test-images".format(len(test_imgs_data)))
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in test_imgs_data]
    print('finish computing hog feature')
    print_duration(start)
    samples = get_hog_descriptors(test_imgs_data)
    #数据降维 reduce the dimension from 3D to 2D
    nsamples, nx, ny = samples.shape
    samples = samples.reshape((nsamples,nx*ny))
    class_labels = get_class_labels(test_imgs_data)
    class_labels = class_labels.ravel()
    del test_imgs_data
    print_duration(start)
    print("Performing batch  classification over all data  ...")
    clf = load(model_path_svm)
    result = clf.score(samples,class_labels)

    print_duration(start)
    print('the test result is: ',result)
    #combine kalman filter
    predict_prob = clf.predict_proba(samples)
    predict_prob = predict_prob.T
    output_kalman = []
    x_state = KalmanFilter()
    for i in range(len(predict_prob[0])):
        #kalman part
        x_state.predict()
        x_state.correct(predict_prob[:,i])

        x_combination = x_state.x_e
        output_kalman.append(np.argmax(x_combination))

    error_kalman, missclass_list_kalman = errorCalculation_return_missclassList(class_labels, output_kalman)
    print("after combining kalman filter the classifier got {}% of the testing examples correct!".format(round((1.0 - error_kalman) * 100, 2)))
    print_duration(start)

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
                img_hog = cv2.resize(img,
                                     (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]),
                                     interpolation=cv2.INTER_AREA)
                hog_descriptor = cv2.HOGDescriptor().compute(img_hog)
                hog_descriptor = hog_descriptor.reshape((1, 3780))
                label = DATA_CLASS_NAMES.get(class_name)
                clf = load(model_path_svm)
                result = clf.predict_proba(hog_descriptor)
                output = result.argmax()
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

################################################################################

if __name__ == '__main__':
    main()


