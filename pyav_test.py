import av
import numpy as np

def read_video(video_path:str) -> np.ndarray:
    # 컨테이너 열기
    container = av.open(video_path)
    # 비디오 스트림 가져오기
    video_stream = container.streams.video[0]

    # 프레임 수, 높이, 너비 계산
    n_frames = video_stream.frames
    height = video_stream.height
    width = video_stream.width

    # numpy ndarray 초기화
    video = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

    # 비디오 프레임 반복문
    for i, frame in enumerate(container.decode(video_stream)):
        # 프레임을 numpy ndarray로 변환
        image = frame.to_image()
        video[i] = np.array(image)

    return video

if __name__ == '__main__':
    # test
    video_path = "/home/longcat2957/Desktop/projects/action_recognition/ViViT-pytorch/data/UCF101/raw/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    video = read_video(video_path)
    print(video.shape)