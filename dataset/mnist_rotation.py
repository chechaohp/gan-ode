import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import numpy as np

class MNISTRotationVideo(Dataset):
    def __init__(self, path_to_data, train = True, N = 500, T = 16, transform = None):
        if os.path.exists(path_to_data):
            data = loadmat(path_to_data)
        else:
            raise FileExistsError(f"File {path_to_data} does not exists")
        self.X = torch.from_numpy(data['X'].squeeze())
        self.Y = torch.from_numpy(data['Y'].squeeze())

        self.transform = transform

        if train == True:
            self.X = self.X[:N].view([N,T,1,28,28])
            self.Y = self.Y[:N]
        else:
            self.X = self.X[N:].view([N,T,1,28,28])
            self.Y = self.Y[N:]

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self,idx):
        video = self.X[idx]
        if self.transform is not None:
            video = self.transform(video)
        return video.float(), self.Y[idx]


class MNISTRotationImage(Dataset):
    def __init__(self, path_to_data, train = True, N = 500, T = 16, transform = None):
        if os.path.exists(path_to_data):
            data = loadmat(path_to_data)
        else:
            raise FileExistsError(f"File {path_to_data} does not exists")
        self.X = torch.from_numpy(data['X'].squeeze())
        self.Y = torch.from_numpy(data['Y'].squeeze())

        self.T = T
        self.transform = transform

        if train == True:
            self.X = self.X[:N].view([N,T,1,28,28])
            self.Y = self.Y[:N]
        else:
            self.X = self.X[N:].view([N,T,1,28,28])
            self.Y = self.Y[N:]

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self,idx):
        #randomly sample image from each video
        random_frame_idx = np.random.randint(0,self.T)
        image = self.X[idx,random_frame_idx,:,:,:]
        if self.transform is not None:
            image = self.transform(image)
        return image.float(), self.Y[idx]
