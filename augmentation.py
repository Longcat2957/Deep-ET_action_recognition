import cv2
import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        """
        주어진 리스트(transforms)에 포함된 변환 함수들을 순서대로 적용하는 Compose 클래스
        """
        self.transforms = transforms
    
    def __call__(self, clips:np.ndarray):
        """
        주어진 클립에 저장된 모든 프레임에 대해, 저장된 모든 변환 함수를 순서대로 적용하여 변환된 클립을 반환
        """
        for t in self.transforms:
            clips = t(clips)
        return clips

class BaseTransform:
    def __init__(self):
        """
        기본적인 변환을 수행하는 BaseTransform 클래스
        """
        pass

    def _transform(self, frame):
        """
        변환 함수를 구현해야 합니다.
        """
        pass

    def __call__(self, clips:np.ndarray):
        """
        입력된 클립에 저장된 모든 프레임에 대해 _transform 메서드를 순서대로 적용하여 변환된 클립을 반환합니다.
        """
        out_clips = []
        for clip in clips:
            out_frames = []
            for frame in clip:
                out_frame = self._transform(frame)
                out_frames.append(out_frame)
            out_clips.append(np.stack(out_frames))
        return np.stack(out_clips)

    
class ResizeTransform(BaseTransform):
    def __init__(self, size):
        """
        입력 프레임의 크기를 조정하는 ResizeTransform 클래스
        :param size: (width, height) 형태의 정수나 튜플
        """
        super().__init__()
        if isinstance(size, int):
            self.width = self.height = size
        elif isinstance(size, tuple):
            self.width, self.height = size
        else:
            raise ValueError("size must be an integer or a tuple of integers")
        
    def _transform(self, frame):
        """
        주어진 프레임의 크기를 조정하여 반환합니다.
        """
        return cv2.resize(frame, (self.width, self.height))
    
class ToTensor():
    def __init__(self, half:bool=False):
        self.dtype = torch.float16 if half else torch.float32

    def __call__(self, clips:np.ndarray):
        clips = clips.astype(np.float32) / 255.0
        clips = np.transpose(clips, (0, 1, 4, 2, 3))
        tensors = torch.from_numpy(clips).to(dtype=self.dtype)
        return tensors
