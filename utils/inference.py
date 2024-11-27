import torch
import librosa
import argparse
import numpy as np
import torch.nn.functional as F
from python_speech_features import logfbank
from argparse import Namespace

import utils as avhubert_utils
from fairseq import checkpoint_utils, utils

def load_audio(audio_path: str, stack_size: int) -> torch.Tensor:
    """
    wav포맷 오디오 파일을 불러와 특성값 추출, 제로 패딩, 텐서 변환 및 정규화 진행.
        
    Args: 
        audio_path: 입력 오디오 파일 경로.
        stack_size: 윈도우 묶음 크기.

    Returns:
        torch.Tensor: 정규화된 log Mel-filterbank energy 특징 텐서. 
                      차원은 (1, stack_size * F, padded(T)/stack_size).
    """
    ## 오디오 파일을 float 값이 담긴 시계열 시퀀스로 로드하기.
    ## 샘플링 레이트를 16,000 Hz 으로 설정하면 초당 16,000개의 샘플을 읽어들인다.
    wav_data, sample_rate = librosa.load(audio_path, sr=16_000)

    ## 오디오 신호로부터 log Mel-filterbank energy 특성 계산하기.
    ## 해당 함수는 주어진 2D 배열을 슬라이딩 윈도우로 분할해서 처리한다.
    ## 기본 윈도우 크기: 400 (16,000Hz에서 25ms는 400 샘플)
    ## 기본 이동 크기: 160 (16,000Hz에서 10ms는 160 샘플)
    ## 각 윈도우에 대해 26개의 주파수 특성 벡터를 생성한다.
    audio_feats = logfbank(wav_data, 
                           samplerate=sample_rate, 
                           winlen=0.025, 
                           winstep=0.01, 
                           nfilt=26).astype(np.float32)

    ## 여기까지 audio_feats의 차원은 (T: 원도우 개수, F: 주파수 특성 개수)
    ## 모델에 입력값으로 제공되는 오디오 데이터는 연속성이 어느정도 보장되어야 한다.
    ## 따라서 stack_size 단위로 윈도우를 묶어서 연산에 활용해야 한다.
    ## 또한, stack_size 기준으로 T가 완벽하게 나누어 떨어져야 데이터 크기가 일관적으로 유지된다.
    ## 따라서 제로 패딩을 가하고, (padded(T)/stack_size, stack_size * F)차원 배열로 변환한다.
    feature_dim = audio_feats.shape[1]
    remainder = len(audio_feats) % stack_size
    if remainder != 0:
        pad_length = stack_size - remainder
        audio_feats = np.pad(audio_feats, ((0, pad_length), (0, 0)), mode='constant')
    audio_feats = audio_feats.reshape((-1, stack_size, feature_dim)).reshape(-1, stack_size * feature_dim)

    ## 오디오 특성값을 텐서로 변환, audio_feats의 0번째 axis 기준으로 정규화.
    audio_feats = torch.from_numpy(audio_feats)
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, normalized_shape=(stack_size * feature_dim,))
    
    ## 모델 입력을 위한 차원 맞추기: (1, stack_size * F, padded(T)/stack_size)
    ## 오디오 파일 하나를 배치 하나에 전부 담아 넘겨주기 위함.
    audio_feats = audio_feats[None,:,:].transpose(1, 2)
    return audio_feats


def load_video(video_path: str, task: object) -> torch.Tensor:
    """
    비디오 파일을 로드하고, 모델에 입력하기 적합한 형태로 전처리한다.

    Args:
        video_path: 입력 비디오 파일 경로.
        task: 비디오 전처리에 필요한 설정값을 포함하는 객체.
            `task.cfg`에는 다음과 같은 설정값이 포함되어 있다:
            - image_crop_size (int): 비디오 프레임을 중앙에서 자를 크기.
            - image_mean (float): 픽셀 값을 정규화할 때 사용할 평균값.
            - image_std (float): 픽셀 값을 정규화할 때 사용할 표준편차값.

    Returns:
        torch.Tensor: [1, 1, T, H, W] 차원으로 전처리된 비디오 프레임 텐서
            - 1: 배치 크기
            - 1: 채널
            - T: 시간 프레임 개수
            - H: 프레임의 높이
            - W: 프레임의 너비
    """
    ## 비디오 ROI 파일에서 프레임을 로드.
    video_feats = avhubert_utils.load_video(video_path)
    ## 비디오 프레임 전처리를 위한 변환(transform) 파이프라인을 정의.
    transform = avhubert_utils.Compose([
        ## 픽셀 값을 [0, 255] 범위에서 [0.0, 1.0] 범위로 정규화.
        avhubert_utils.Normalize(0.0, 255.0),
        ## 비디오 프레임의 중심을 기준으로 설정된 크기로 자른다.
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        ## 픽셀 값을 평균과 표준편차로 정규화.
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])

    ## 정의한 변환을 비디오 프레임에 적용.
    video_feats = transform(video_feats)
    ## 변환된 비디오 데이터를 PyTorch 텐서로 변환.
    video_feats = torch.FloatTensor(video_feats)
    ## 모델 입력 형식에 맞추기 위해 배치 차원, 채널 차원 추가.
    video_feats = video_feats.unsqueeze(dim=0).unsqueeze(dim=0)
    return video_feats


def extract_feature(checkpoint_path: str,
                    video_path: str,
                    audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    오디오 및 비디오 데이터를 입력으로 받아, AV-HuBERT를 사용해 특징 벡터를 추출.

    Args:
        checkpoint_path: 사전 학습된 모델의 체크포인트 파일 경로.
        video_path: 입력 비디오 파일 경로.
        audio_path: 입력 오디오 파일 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - feature_audio (np.ndarray): 추출된 오디오 특징 벡터.
            - feature_video (np.ndarray): 추출된 비디오 특징 벡터.
    """
    ## 디바이스 설정.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##
    utils.import_user_module(Namespace(user_dir="av_hubert/avhubert"))

    ## 체크포인트에서 AV-HuBERT 인코더와 태스크 로드.
    checkpoint, _, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = checkpoint[0].encoder.w2v_model

    ## 모델 디바이스 보내고 평가 모드로 설정.
    model.to(device)
    model.eval()

    ## ROI 비디오, 오디오 로드.
    video_feats = load_video(video_path, task=task)
    audio_feats = load_audio(audio_path, stack_size=4)

    ## 오디오와 비디오 텐서 타임스텝 동기화.
    residual = video_feats.shape[2] - audio_feats.shape[-1]
    if residual > 0:
        video_feats = video_feats[:, :, :-residual]
    elif residual < 0:
        audio_feats = audio_feats[:, :, :residual]

    ## 동기화된 비디오, 오디오 텐서 디바이스로 올리기.
    video_feats = video_feats.to(device)
    audio_feats = audio_feats.to(device)
    
    ## AV-HuBERT 인코더를 사용해 오디오와 비디오의 특징 벡터를 추출.
    with torch.no_grad():
        feature_audio, _ = model.extract_finetune(
            source={'video': None, 'audio': audio_feats},
            padding_mask=None,
            output_layer=None)
        feature_audio = feature_audio.squeeze(dim=0)
        feature_video, _ = model.extract_finetune(
            source={'video': video_feats, 'audio': None},
            padding_mask=None,
            output_layer=None)
        feature_video = feature_video.squeeze(dim=0)
    return feature_audio.cpu().numpy(), feature_video.cpu().numpy()


def evaluate(feature_video: np.ndarray,
             feature_audio: np.ndarray) -> float:
    """
    """
    ## 유사도 점수를 저장할 리스트
    similarity_scores = []

    ## 비디오, 오디오 특징 벡터의 각 프레임에 대해 계산
    for i in range(feature_audio.shape[0]):
        # 벡터 추출
        feature_audio_vec = feature_audio[i]
        feature_video_vec = feature_video[i]
        # 벡터 정규화
        feature_audio_unit = feature_audio_vec / np.linalg.norm(feature_audio_vec)
        feature_video_unit = feature_video_vec / np.linalg.norm(feature_video_vec)
        # 코사인 유사도 계산
        similarity = np.dot(feature_audio_unit, feature_video_unit)
        similarity_scores.append(similarity)

    ## 0.03 분위수 값 추출
    lambda_percentile = 3
    final_score = np.percentile(similarity_scores, lambda_percentile)
    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy', type=str, default=None, help='A dummy argument for special cases')
    parser.add_argument('-c', '--check_path', type=str, default="misc/model.pt", required=False)
    parser.add_argument('-v', '--video_path', type=str, default="data/roi/002_mouth_roi.mp4", required=False)
    parser.add_argument('-a', '--audio_path', type=str, default="data/audio/002.wav", required=False)
    args = parser.parse_args()

    feature_video, feature_audio = extract_feature(
        checkpoint_path=args.check_path,
        video_path=args.video_path,
        audio_path=args.audio_path)

    score = evaluate(feature_video, feature_audio)
    print(score)