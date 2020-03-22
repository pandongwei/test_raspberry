from utils import *
from joblib import load
import cv2
from kalman_filter import *
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

def main():

    #测试部分开始
    image_path = '/home/pi/master-thesis/dataset-sample-test'
    # image_path = '/home/pan/master-thesis-in-mrt/data/dataset-test'
    model_path_svm = 'train_result/traditional_method/hog_svm_whole_83.8%_85.12%_87.98%.joblib'
    model_path_bagging = 'train_result/traditional_method/train_whole_dataset_bagging_83.42%_85.69%_88.57%.joblib'

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

################################################################################

if __name__ == '__main__':
    main()


