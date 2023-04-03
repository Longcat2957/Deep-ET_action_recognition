import os
import av
from typing import Tuple, List, Optional, Callable
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from augmentation import Compose, ResizeTransform, ToTensor

"""
    UCF101 데이터셋은 101개의 액션 카테고리를 가지는 비디오 데이터셋입니다.
    데이터셋은 13,320개의 비디오 클립으로 구성되어 있으며,
    각 클립의 길이는 평균 6.5초이고,
    최대 길이는 735프레임,
    최소 길이는 28프레임입니다.
    이러한 비디오 클립들은 25fps의 프레임 레이트를 가지며,
    해상도는 320x240 또는 640x480로 제공됩니다.
"""

def read_classInd(p:str) -> Tuple[dict, dict]:
    name2idx = {}
    idx2name = {}
    with open(p, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        _, name = line.split()
        name2idx[name] = idx
        idx2name[idx] = name
    return name2idx, idx2name

def split_list(lst: List, ratios: Tuple[float, float, float], seed: int = None) -> Tuple[List, List, List]:
    assert len(lst) > 1, "Input list should have at least 2 elements."
    total = len(lst)
    train_len = max(1, int(total * ratios[0]))
    val_len = max(1, int(total * ratios[1]))
    test_len = max(1, total - train_len - val_len)
    assert train_len + val_len + test_len == total, "Ratios should add up to 1.0"

    if seed is not None:
        random.seed(seed)
    random.shuffle(lst)

    train_list = lst[:train_len]
    val_list = lst[train_len:train_len+val_len]
    test_list = lst[train_len+val_len:]

    return train_list, val_list, test_list

def split_data(root:str, ratio:Tuple[float, float, float]=(0.8, 0.1, 0.1)) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    train_files, valid_files, test_files = [], [], []
    for name in os.listdir(root):
        class_dir = os.path.join(root, name)
        class_dir = [os.path.join(class_dir, x) for x in os.listdir(class_dir)]
        train_list, val_list, test_list = split_list(class_dir, ratio)
        train_files.extend(train_list)
        valid_files.extend(val_list)
        test_files.extend(test_list)
    return train_files, valid_files, test_files

class UCF101_Dataset(Dataset):
    def __init__(
            self,
            data_list:list,
            annotation_dicts:str,
            frames_per_clip:int=16,
            step_between_clips:int=2,
            transform:Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.name2idx, self.idx2name = read_classInd(annotation_dicts)
        # videos abs path
        self.data_list = data_list
        self.class_list = self._make_class_list()
        # number of classes
        self.nc = 101
        
        # *.avi file -> T, C, H, W np.ndarray
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips

        # To Do, transform
        if transform is None:
            self.transform = Compose([
                ResizeTransform(224),
                ToTensor()
            ])
    
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_tensor = self._one_hot_encoding(self.class_list[idx])
        video_array = self._read_video(self.data_list[idx])
        
        number_of_clips, video_clips = self._extract_clips(video_array)
        target_tensor = torch.tile(target_tensor, (number_of_clips, 1))

        # after transform ...
        video_tensor = self.transform(video_clips)
        # convert 0~255 -> 0.0 ~ 1.0 and np.ndarray -> torch.Tensor
        # video_clips = video_clips.astype(np.float32) / 255.0
        # video_clips = np.transpose(video_clips, (0, 1, 4, 2, 3))
        # video_tensor = torch.from_numpy(video_clips)

        return video_tensor, target_tensor

    def _make_class_list(self) -> list:
        class_list = []
        for abs_path in self.data_list:
            class_name = abs_path.split('/')[-2]
            class_idx = self.name2idx[class_name]
            class_list.append(class_idx)
        return class_list

    def _one_hot_encoding(self, class_num:int) -> torch.Tensor:
        target_tensor = torch.zeros(101)
        target_tensor[class_num] = 1
        return target_tensor.float()

    def _read_video(self, p:str) -> torch.Tensor:
        # 컨테이너 열기, 비디오 스트림 가져오기
        container = av.open(p)
        video_stream = container.streams.video[0]

        # 프레임 수, 높이, 너비를 계산
        n_frames = video_stream.frames
        height = video_stream.height
        width = video_stream.width

        # numpy.ndarray 초기화
        video = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

        # 비디오 프레임 반복문
        for i, frame in enumerate(container.decode(video_stream)):
            # 프레임을 np.ndarray로 변환
            image = frame.to_image()
            video[i] = np.array(image)
        return video
    
    def _extract_clips(self, v_arr:np.ndarray) -> np.ndarray:
        T, H, W, C = v_arr.shape
        
        # 프레임 간격(step) 조정
        step = self.frames_per_clip + (self.step_between_clips - 1) * self.frames_per_clip

        # 클립 수 계산
        num_clips = (T - step) // step + 1

        # numpy ndarray 초기화
        clips = np.zeros((num_clips, self.frames_per_clip, H, W, C), dtype=np.uint8)

        # 비디오 클립 반복문
        for i in range(num_clips):
            # 클립 시작 프레임과 끝 프레임 인덱스를 계산
            start_frame = i * self.step_between_clips
            end_frame = start_frame + self.frames_per_clip

            clips[i] = v_arr[start_frame:end_frame]
        return num_clips, clips

    @staticmethod
    def collate_fn(batch):
        video_tensor, labels = zip(*batch)
        return torch.cat(video_tensor, 0), torch.cat(labels, 0)

if __name__ == '__main__':
    train, val, test = split_data("../data/UCF101/raw")
    test_dataset = UCF101_Dataset(train, "../data/UCF101/classInd.txt")
    v, c = test_dataset[1230]
    print(f"video_clips_shape = {v.shape}, {c.shape}")

    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=16, collate_fn=test_dataset.collate_fn)

    for v, c in loader:
        print(v.shape, c.shape)
        break

    from vivit import ViViT
    net = ViViT(224, 16, 101, 16)
    o = net(v)
    print(o.shape)