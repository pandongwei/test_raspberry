from keras.preprocessing.image import ImageDataGenerator
from utils import *
import time
from model.mobilenet_v3_large import MobileNetV3_Large



def generate(batch, shape, test):
    """Data generation and augmentation
    """
    datagen1 = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen1.flow_from_directory(
        test,
        target_size=shape,
        batch_size=batch,
        shuffle=True,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(test):
        for each in files:
            count1 += 1

    return test_generator, count1


def train():
    weight_path = ''
    image_path = '/home/pi/master-thesis/dataset-sample-test'
    batch_size = 4
    model = MobileNetV3_Large((224,224,3), 4).build()
    model.load_weights(weight_path, by_name=True)

    # test the result
    train_generator, count1 = generate(batch_size, (224,224),image_path)
    start = time.time()
    result = model.evaluate_generator(train_generator,use_multiprocessing=True)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    train()
