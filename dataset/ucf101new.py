from io import FileIO
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from dataset.video.video_utils import read_video
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import numpy as np

import os

class UCF101Video(Dataset):
    def __init__(self, 
                root:str, 
                video_folder:str, 
                annotation_folder:str,
                n_frame:int = 16,
                train:bool = True,
                fold:int = 1,
                image_size:int = 64):
        super().__init__()
        assert fold in [1,2,3], f"fold need to have value 1,2 or 3, not {fold}"
        self.video_folder = os.path.join(root,video_folder)
        self.annotation_folder = os.path.join(root, annotation_folder)
        self.train = train
        self.n_frame = n_frame
        self.fold = fold
        self.image_size = image_size
        # set image size to 64 x 85
        self.transform = Resize((64,85), InterpolationMode.BICUBIC)
        self.classes, self.class_to_idx = self.find_classes()
        self.samples = self.make_dataset()

    def find_classes(self) -> Tuple[List[str],Dict[str,int,int]]:
        # get path of label file
        f = os.path.join(self.annotation_path,'classInd.txt')
        class_to_idx={}
        with open(f,"r") as class_file:
            data = class_file.readlines()
            data = [x.split(" ") for x in data]
            data = [[int(x[0]),x[1].replace('\n','')] for x in data]
            classes = [x[1] for x in data]
            for line in data:
                class_to_idx[line[1]] = line[0]
        return classes, class_to_idx


    def make_dataset(self) -> List[Tuple[str,int]]:
        data = "train" if self.train else "test"
        annotation_file = os.path.join(self.annotation_folder, f"{data}list0{self.fold}")
        # read content
        with open(annotation_file, "r") as file:
            lines = file.readlines()
        
        lines = [line.split()[0] for line in lines]
        video_path = []
        # get only the video in index file
        for path in lines:
            _class = path.split("/")[0]
            if _class in self.classes:
                video, sound, meta = read_video(path)
                length = int(video.size()[0])
                video_path.append((path, length, self.class_to_idx[_class]))
        return video_path
    
    def __len__(self):
        return len(self.samples)
    
    def resize_crop(self,x):
        x = self.transform(x)
        x = x[:, :, :, 10:10 + self.img_size]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def __getitem__(self, index):
        video_path, length,_class = self.samples[index]
        start_frame = np.random.randint(length - self.n_frame)
        end_frame = start_frame + self.n_frame - 1
        video, sound, meta = read_video(video_path, start_frame, end_frame)
        del sound
        del meta
        video = video.permute(0,3,1,2)
        video = self.resize_crop(video)
        video = (video - 128.0)/128.0
        return video, _class

class UCF101Video(UCF101Video):
    def __init__(self, 
                root:str, 
                video_folder:str, 
                annotation_folder:str,
                n_frame:int = 16,
                train:bool = True,
                fold:int = 1,
                image_size:int = 64):
        super().__init__(root, video_folder, annotation_folder,n_frame,train,fold,image_size)

    def resize_crop(self, x):
        x = self.transform(x)
        x = x[:, :, 10:10 + self.img_size]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def __getitem__(self, index):
        video_path, length, _class = self.samples[index]
        random_idx = np.random.randint(length)
        video, sound, meta = read_video(video_path)
        del sound
        del meta
        image = video[random_idx]
        image = image.permute(2,0,1)
        image = self.resize_crop(image)
        image = (image - 128.0)/128.0
        return image, _class