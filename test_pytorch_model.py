import cv2
from model.mobilenetv2_4channels import mobilenetv2_4channels
from model.mobilenetv2_3d import get_model
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset

DATA_CLASS_NAMES = {
    "bicycle-lane":0,
    "bicycle-lane-and-pedestrian":1,
    "car-lane":2,
    "pedestrian":3
}

class SequentialDataset(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path,  height = 224, width = 224,rescale = None):
        self.root_path = root_path
        self.fnames, self.labels = [], []
        self.height = height
        self.width = width
        self.rescale = rescale
        self.alpha = 0.5
        self.inv = False

        for label in sorted(os.listdir(root_path)):
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                self.labels.append(DATA_CLASS_NAMES.get(label))
                self.fnames.append(os.path.join(root_path, label, fname))
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        frame = np.array(cv2.imread(self.fnames[index])).astype(np.float64)
        labels = np.array(self.labels[index])
        tmp = frame.copy()
        if self.rescale is not None:
            frame = frame*self.rescale
        extra_channel = maddern2014(tmp,alpha=self.alpha,inv=self.inv) # rescale is already included in this function
        frame = np.concatenate((frame,extra_channel),axis=2)
        frame = self.to_tensor(frame)  #更换维度的顺序
        return torch.from_numpy(frame), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((2,0,1))

class SequentialDataset_sequence(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path, images_len=10,  height=224, width=224, rescale=None):
        self.root_path = root_path
        self.images_len = images_len
        self.fnames, self.labels = [], []
        self.height = height
        self.width = width
        self.rescale = rescale
        part = []
        for label in sorted(os.listdir(root_path)):
            i = 0
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                if i < self.images_len:
                    part.append(os.path.join(root_path, label, fname))
                    i += 1
                else:
                    self.labels.append(DATA_CLASS_NAMES.get(label))
                    self.fnames.append(part)
                    part = []
                    i = 0
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        buffer = np.empty((self.images_len, self.height, self.width, 3), np.dtype('float32'))
        for i,frame_name in enumerate(self.fnames[index]):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            if i < self.images_len:
                buffer[i] = frame
            else:
                break
        labels = np.array(self.labels[index])
        if self.rescale is not None:
            buffer = buffer*self.rescale
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))


def maddern2014(image,alpha,inv=False):
    """
    Implement the algorithm proposed by Will Maddern et al. in ICRA2014
    Paper:
    Illumination Invariant Imaging: Applications in Robust Vision-based
    Localisation, Mapping and Classification for Autonomous Vehicles

    ii_image = rgb2ii.maddern2014(image, alpha, inv)
    where
    image : color image data
    alpha : a camera-dependent parameter ranged in 0-1
    inv   : perform image inversion (a=1-a) if inv is true (default: false)
    """
    image = image / 255.
    #ii_image = 0.5 + math.log(image[:,:,1]) - alpha*math.log(image[:,:,2]) - (1-alpha)*math.log(image[:,:,0])
    ii_image = np.full((224,224),0.5) + np.log(image[:, :, 1] + 1e-10) - alpha * np.log(image[:, :, 2] + 1e-10) - (1 - alpha) * np.log(
        image[:, :, 0] + 1e-10)
    if inv:
        ii_image = 1-ii_image
    return ii_image[:,:,np.newaxis]


def main():
    test_model = 'm2'  #m2 / 3d-m2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = '/home/pi/master-thesis/dataset-test'
    #image_path = '/home/pan/master-thesis-in-mrt/data/dataset-test'
    batch_size = 4
    if test_model =='m2':
        weight_path = 'train_result/network/MobileNetV2_4channels_model_final_93.57%.pkl'
        model = mobilenetv2_4channels(pretrained=False,n_class=4)
        test_dataloader = DataLoader(SequentialDataset(root_path=image_path, rescale=1 / 255.),
                                     batch_size=batch_size, num_workers=4)
    elif test_model == '3d-m2':
        weight_path = 'train_result/network/3DM2_model_final_87.01%.pkl'
        model = get_model(num_classes=4, sample_size=224, width_mult=1.0)
        test_dataloader = DataLoader(SequentialDataset_sequence(root_path=image_path, rescale=1 / 255.),
                                     batch_size=batch_size, num_workers=4)

    weights = torch.load(weight_path,map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model = model.to(device)

    #test the result
    running_corrects = 0.0
    test_size = len(test_dataloader.dataset)
    print("testing with {} test-images".format(test_size))
    model.eval()
    start = time.time()
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device,dtype=torch.float)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(test_size)
    print(running_corrects)
    epoch_acc = running_corrects / test_size
    print(epoch_acc)

if __name__ == '__main__':
    main()
