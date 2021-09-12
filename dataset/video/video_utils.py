import bisect
from fractions import Fraction
import math
import torch
from torchvision.io import (
    _read_video_timestamps_from_file,
    _read_video_from_file,
    _probe_video_from_file
)
from torchvision.io import read_video_timestamps
from torchvision.io.video import _check_av_available, _align_audio_frames
from torchvision.io import _video_opt
import gc
import re
import warnings
import numpy as np


from torch.utils.model_zoo import tqdm


try:
    import av
    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, 'pict_type'):
        av = ImportError("""\
Your version of PyAV is too old for the necessary video operations in torchvision.
If you are on Python 3.5, you will have to build from source (the conda-forge
packages are not up-to-date).  See
https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")
except ImportError:
    av = ImportError("""\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")


# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

def _read_from_stream(container, start_offset, end_offset, pts_unit, stream, stream_name):
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    if pts_unit == 'sec':
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results and will be removed in a " +
                      "follow-up version. Please use pts_unit 'sec'.")

    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(br"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(br"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    try:
        for idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        # TODO add a warning
        pass
    # ensure that the results are sorted wrt the pts
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result


def read_video(filename, start_pts=0, end_pts=None, pts_unit='pts'):
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames
    Parameters
    ----------
    filename : str
        path to the video file
    start_pts : int if pts_unit = 'pts', optional
        float / Fraction if pts_unit = 'sec', optional
        the start presentation time of the video
    end_pts : int if pts_unit = 'pts', optional
        float / Fraction if pts_unit = 'sec', optional
        the end presentation time
    pts_unit : str, optional
        unit in which start_pts and end_pts values will be interpreted, either 'pts' or 'sec'. Defaults to 'pts'.
    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[K, L]
        the audio frames, where `K` is the number of channels and `L` is the
        number of points
    info : Dict
        metadata for the video and audio. Can contain the fields video_fps (float)
        and audio_fps (int)
    """

    from torchvision import get_video_backend
    if get_video_backend() != "pyav":
        return _video_opt._read_video(filename, start_pts, end_pts, pts_unit)

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))

    info = {}
    video_frames = []
    audio_frames = []

    try:
        container = av.open(filename, metadata_errors='ignore')
    except av.AVError:
        # TODO raise a warning?
        pass
    else:
        if container.streams.video:
            video_frames = _read_from_stream(container, start_pts, end_pts, pts_unit,
                                             container.streams.video[0], {'video': 0})
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(container, start_pts, end_pts, pts_unit,
                                             container.streams.audio[0], {'audio': 0})
            info["audio_fps"] = container.streams.audio[0].rate

        container.close()

    vframes = [frame.to_rgb().to_ndarray() for frame in video_frames]
    aframes = [frame.to_ndarray() for frame in audio_frames]

    if vframes:
        vframes = torch.as_tensor(np.stack(vframes))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes:
        aframes = np.concatenate(aframes, 1)
        aframes = torch.as_tensor(aframes)
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    return vframes, aframes, info


def pts_convert(pts, timebase_from, timebase_to, round_func=math.floor):
    """convert pts between different time bases
    Args:
        pts: presentation timestamp, float
        timebase_from: original timebase. Fraction
        timebase_to: new timebase. Fraction
        round_func: rounding function.
    """
    new_pts = Fraction(pts, 1) * timebase_from / timebase_to
    return round_func(new_pts)


def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


class _DummyDataset(object):
    """
    Dummy dataset used for DataLoader in VideoClips.
    Defined at top level so it can be pickled when forking.
    """
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return read_video_timestamps(self.x[idx])


class VideoClips(object):
    """
    Given a list of video files, computes all consecutive subvideos of size
    `clip_length_in_frames`, where the distance between each subvideo in the
    same video is defined by `frames_between_clips`.
    If `frame_rate` is specified, it will also resample all the videos to have
    the same frame rate, and the clips will refer to this frame rate.

    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.

    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.

    Arguments:
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        frames_between_clips (int): step (in frames) between each clip
        frame_rate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the clips will be defined
            on the resampled video
        num_workers (int): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1,
                 frame_rate=None, _precomputed_metadata=None, num_workers=0,
                 _video_width=0, _video_height=0, _video_min_dimension=0,
                 _audio_samples=0, _audio_channels=0):

        self.video_paths = video_paths
        self.num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._audio_samples = _audio_samples
        self._audio_channels = _audio_channels

        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _collate_fn(self, x):
        return x

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []

        # strategy: use a DataLoader to parallelize read_video_timestamps
        # so need to create a dummy dataset first
        import torch.utils.data
        dl = torch.utils.data.DataLoader(
            _DummyDataset(self.video_paths),
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn)

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

    def _init_from_metadata(self, metadata):
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

    @property
    def metadata(self):
        _metadata = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps
        }
        return _metadata

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps
        }
        return type(self)(video_paths, self.num_frames, self.step, self.frame_rate,
                          _precomputed_metadata=metadata, num_workers=self.num_workers,
                          _video_width=self._video_width,
                          _video_height=self._video_height,
                          _video_min_dimension=self._video_min_dimension,
                          _audio_samples=self._audio_samples,
                          _audio_channels=self._audio_channels)

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_frames, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.

        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        from torchvision import get_video_backend
        backend = get_video_backend()

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_min_dimension != 0")
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video, audio, info = read_video(video_path, start_pts, end_pts)
        else:
            info = _probe_video_from_file(video_path)
            video_fps = info["video_fps"]
            audio_fps = None

            video_start_pts = clip_pts[0].item()
            video_end_pts = clip_pts[-1].item()

            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase = Fraction(0, 1)
            if "audio_timebase" in info:
                audio_timebase = info["audio_timebase"]
                audio_start_pts = pts_convert(
                    video_start_pts,
                    info["video_timebase"],
                    info["audio_timebase"],
                    math.floor,
                )
                audio_end_pts = pts_convert(
                    video_end_pts,
                    info["video_timebase"],
                    info["audio_timebase"],
                    math.ceil,
                )
                audio_fps = info["audio_sample_rate"]
            video, audio, info = _read_video_from_file(
                video_path,
                video_width=self._video_width,
                video_height=self._video_height,
                video_min_dimension=self._video_min_dimension,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase=info["video_timebase"],
                audio_samples=self._audio_samples,
                audio_channels=self._audio_channels,
                audio_pts_range=(audio_start_pts, audio_end_pts),
                audio_timebase=audio_timebase,
            )

            info = {"video_fps": video_fps}
            if audio_fps is not None:
                info["audio_fps"] = audio_fps

        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, "{} x {}".format(video.shape, self.num_frames)
        return video, audio, info, video_idx
