import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from utils import load_value_file

from typing import List, Callable, Dict, Tuple


def pil_loader(path: str) -> Image:
    """ Read RGB image using PIL
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path: str):
    """ Load image using accimage
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    """ Choose get default image loader from torchvision backend
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path: str, frame_indices: List[int], image_loader: Callable) -> List:
    """ load video
    """
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video



def get_default_video_loader() -> Callable:
    """ Get the default video loader
    """
    # get image loader
    image_loader = get_default_image_loader()
    # get video loader with default image loader as params
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path: str) -> Dict:
    """ Load anotation data
    """
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data: Dict) -> Dict:
    """ create a label to index mapping 
    """
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map
    

def get_video_names_and_annotations(data: Dict, subset:str):
    """ Get video information
    """
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path: str, annotation_path: str, subset: str, n_samples_for_each_video: int,
                 sample_duration: int) -> Tuple[List[Dict], Dict[int,str]]:
    """ create dataset
    Args:
        root_path: str
            path to root data folder
        annotation_path: str
            path to annotation data
        subset: 

        n_samples_for_each_videos: int
            number of sample to sample from each video
        sample_duration: int
            number of frame sample from each video
    Returns:
        dataset: List[Dict]
            A list contains all metadata of sample
        idx_to_class: Dict[int,str] 
            A dictionary map from an index of class to class name
    """
    # load annotation data
    data = load_annotation_data(annotation_path)
    # get video name and annotation
    video_names, annotations = get_video_names_and_annotations(data, subset)
    # create a dictionary to map label to index
    class_to_idx = get_class_labels(data)
    # create an inverse maping from index to class
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        # tracking loading process
        if i % 1000 == 0:
            print(f'dataset loading [{i}/{len(video_names)}]')
        # check if video exists
        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue
        # get n_frames value 
        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue
        # create metadata for sample
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            #### For testing: restrict the number of classes
            try:
                sample['label'] = class_to_idx[annotations[i]['label']]
            except:
                continue
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            # when only take one sample from each video
            # get a list of sample index
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                # when sample more than one sample from each video
                # get step for sampling
                step = max(1, math.ceil((n_frames - 1 - sample_duration) / (n_samples_for_each_video - 1)))
            else:
                # if the n_sample is less than 1
                step = sample_duration
            for j in range(1, n_frames, step):
                # create a copy of sample for each step 
                sample_j = copy.deepcopy(sample)
                # get list of sample index
                sample_j['frame_indices'] = [range(j, min(n_frames + 1, j + sample_duration))]
                dataset.append(sample_j)

    return dataset, idx_to_class


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path: str,
                 annotation_path: str,
                 subset: str,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # get path to video
        path = self.data[index]['video']
        # get frame indeces
        frame_indices = self.data[index]['frame_indices']
        # do temporal transform
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        # load clip
        clip = self.loader(path, frame_indices)
        # do spatial transform
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)