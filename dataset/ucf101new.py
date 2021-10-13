from io import FileIO
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from dataset.video.video_utils import read_video
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
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

    def find_classes(self) -> Tuple[List[str],Dict[str,int]]:
        # get path of label file
        f = os.path.join(self.annotation_folder,'classInd.txt')
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
        annotation_file = os.path.join(self.annotation_folder, f"{data}list0{self.fold}.txt")
        # read content
        with open(annotation_file, "r") as file:
            lines = file.readlines()
        
        lines = [line.split()[0] for line in lines]
        video_path = []
        # get only the video in index file
        for path in tqdm(lines):
            _class = path.split("/")[0]
            complete_path = os.path.join(self.video_folder,path)
            if _class in self.classes:
                video, sound, meta = read_video(complete_path)
                length = int(video.size()[0])
                if length < self.n_frame:
                    continue
                video_path.append((complete_path, length, self.class_to_idx[_class]))
        return video_path
    
    def __len__(self):
        return len(self.samples)
    
    def resize_crop(self,x):
        x = self.transform(x)
        x = x[:, :, :, 10:10 + self.image_size]
        assert x.shape[2] == self.image_size
        assert x.shape[3] == self.image_size
        return x

    def __getitem__(self, index):
        video_path, length,_class = self.samples[index]
        # print(video_path, length)
        # start_frame = np.random.randint(0,length - self.n_frame-1)
        # end_frame = start_frame + self.n_frame - 1
        # video, sound, meta = read_video(video_path, start_frame, end_frame)
        # video = video.permute(0,3,1,2)
        while (True):
            start_frame = np.random.randint(0,length - self.n_frame-1)
            end_frame = start_frame + self.n_frame - 1
            video, sound, meta = read_video(video_path, start_frame, end_frame)
            video = video.permute(0,3,1,2)
            if video.size(0) == self.n_frame:
                break
        video = self.resize_crop(video)
        video = (video - 128.0)/128.0
        del sound
        del meta
        return video, _class

class UCF101Image(Dataset):
    def __init__(self, 
                root:str, 
                video_folder:str, 
                annotation_folder:str,
                samples = None,
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
        if samples is None:
            self.classes, self.class_to_idx = self.find_classes()
            self.samples = self.make_dataset()
        else:
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def find_classes(self) -> Tuple[List[str],Dict[str,int]]:
        # get path of label file
        f = os.path.join(self.annotation_folder,'classInd.txt')
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
        annotation_file = os.path.join(self.annotation_folder, f"{data}list0{self.fold}.txt")
        # read content
        with open(annotation_file, "r") as file:
            lines = file.readlines()
        
        lines = [line.split()[0] for line in lines]
        video_path = []
        # get only the video in index file
        for path in tqdm(lines):
            _class = path.split("/")[0]
            complete_path = os.path.join(self.video_folder,path)
            if _class in self.classes:
                video, sound, meta = read_video(complete_path)
                length = int(video.size()[0])
                video_path.append((complete_path, length, self.class_to_idx[_class]))
        return video_path

    def resize_crop(self, x):
        x = self.transform(x)
        x = x[:, :, 10:10 + self.image_size]
        assert x.shape[1] == self.image_size
        assert x.shape[2] == self.image_size
        return x

    def __getitem__(self, index):
        video_path, length, _class = self.samples[index]
        # print(video_path, length)
        random_idx = np.random.randint(0,length)
        video, sound, meta = read_video(video_path)
        del sound
        del meta
        image = video[random_idx]
        image = image.permute(2,0,1)
        image = self.resize_crop(image)
        image = (image - 128.0)/128.0
        return image, _class