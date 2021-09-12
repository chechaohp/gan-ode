from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
import os
# from typing import Optional, Dict, Tuple, Callable, List, cast


class UCF101Video(Dataset):
    def __init__(self, 
                 root:str,
                 video_folder:str,
                 annotation_folder:str,
                 frames_per_clip:int,
                 step_between_clips=1,
                 frame_rate=None, 
                 fold=1, 
                 train=True, 
                 transform=None,
                 _precomputed_metadata=None, 
                 num_workers=1, 
                 _video_width=0,
                 _video_height=0, 
                 _video_min_dimension=0, 
                 _audio_samples=0):
        
        super().__init__()

        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.video_folder = os.path.join(root,video_folder)
        self.annotation_path = os.path.join(root, annotation_folder)
        # set the dataset to train dataset or test dataset
        self.train = train
        # get classes name and class_to_edx
        self.classes, self.class_to_idx = self.find_classes()
        print(self.class_to_idx)
        self.samples = make_dataset(self.video_folder, self.class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        # we bookkeep the full version of video clips because we want to be able
        # to return the meta data of full version rather than the subset version of
        # video clips
        self.full_video_clips = video_clips
        self.indices = self.select_fold(video_list, self.annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    
    def find_classes(self):
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
    
    def select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.video_folder, x[0]) for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]
        video = video.permute(0,3,1,2)
        video = [ToPILImage()(image) for image in video]
        if self.transform is not None:
            self.transform.randomize_parameters()
            video = [self.transform(image) for image in video]
            

        return video, label