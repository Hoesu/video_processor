# Video ROI Extractor!

<p align="center">
  <img alt="original" src="assets/original.gif" width="100%">
</p>

<p align="center">
  Original Video
</p>

| ![GIF 1](assets/eye.gif)          | ![GIF 2](assets/nose.gif)          | ![GIF 3](assets/mouth.gif)          |
|-----------------------------------|------------------------------------|-------------------------------------|
| <div align="center">Eye ROI</div> | <div align="center">Nose ROI</div> | <div align="center">Mouth ROI</div> |


## Environment

```bash
  VESSL CONTAINER: PyTorch 2.3.1 (CUDA 12.1)
```

## Installation

```bash
  chmod +x setup.sh
  ./setup.sh
```

## Configuration
```bash
  ## Directory Management
  video_directory: 'data/video'
  audio_directory: 'data/audio'
  roi_directory: 'data/roi'
  mean_face_path: "misc/20words_mean_face.npy"
  face_predictor_path: "misc/shape_predictor_68_face_landmarks.dat"


  ## ROI Settings
  ##
  ##      18    19    20               23    24    25
  ##  17                   21      22                   26
  ##  
  ##          37 38            27           43 44
  ##       36       39         28        42       45
  ##          41 40            29           47 46
  ##                           30
  ##  0                    31      35                  16 
  ##                        32 33 34
  ##   1                                              15
  ##                  
  ##                        50 51 52
  ##     2               49          53              14
  ##                        61 62 63
  ##       3         48  60          64  54        13
  ##        4               67 66 65              12
  ##         5           59          55          11
  ##                        58 57 56            
  ##            6                             10
  ##                7                     9
  ##                           8
  ##
  ## 프리셋 중 하나를 골라서 사용할 수 있습니다.
  ## mouth, nose, right eye, right cheek, right eyebrow, left eye, left cheek, left eyebrow,
  roi_target: "right eye"
  ## 새로운 커스텀 프리셋을 만들어 사용할 수 있습니다.
  ## 원하는 인덱스들을 골라서 리스트 안에 담아주세요.
  ## 기존 프리셋을 사용하기 원하는 경우, 리스트를 비워두세요.
  custom_target: []


  ## Video Landmark Options
  skip_frames: 10
  resized_frame_height: 100
  resized_frame_width: 200


  ## Video Crop Options
  stablePntsIDs: [33, 36, 39, 42, 45]
  std_size: [256, 256]
  crop_height: 96
  crop_width: 96
  window_margin: 12
```

## Deployment

```bash
  python utils/preprocess.py -c config.yaml
```