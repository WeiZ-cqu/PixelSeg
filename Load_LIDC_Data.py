import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import random
import pickle

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None, anno1=False):
        self.transform = transform
        self.anno1 = anno1
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        #image = np.expand_dims(self.images[index], axis=0)
        image = self.images[index]
        #image = cv2.resize(image, (64, 64))
        #labels = [(cv2.resize(self.labels[index][i], (64, 64)) > 0.5).astype(np.float) for i in range(4)]
        #labels = [(self.labels[index][i] > 0.5).astype(np.float) for i in range(4)]
        labels = [self.labels[index][i] for i in range(4)]

        #Randomly select one of the four labels for this image
        if self.anno1:
            label = labels[0].astype(float)
        else:
            label = labels[random.randint(0,3)].astype(float)
#            while np.sum(label) == 0:
#                label = labels[random.randint(0,3)].astype(float)
        sub_label = (cv2.resize(label, (64, 64)) > 0.5).astype(np.float)
        
        
        all_label = np.stack(labels)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        sub_label = torch.from_numpy(sub_label)
        all_label = torch.from_numpy(all_label)

        image = torch.unsqueeze(image, 0)
        label = torch.unsqueeze(label, 0)
        sub_label = torch.unsqueeze(sub_label, 0)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        sub_label = sub_label.type(torch.FloatTensor)
        all_label = all_label.type(torch.FloatTensor)

        return image, label, all_label, sub_label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    