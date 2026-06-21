# 고급파이썬프로그래밍밎자동화 최적화 프로젝트 (Advanced Python Programming)

저장소 구조

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── before/
│   │   └── DiffusionAD_baseline.py        # 최적화 적용 전 원본 훈련 스크립트
│   └── after/
│       └── DiffusionAD_optimized.py  # 최적화가 모두 적용된 제출용 스크립트
├── benchmark/
│   ├── DiffusionAD_baseline_profiled.py     # 최적화 전 성능 측정용 벤치마크 코드
│   └── DiffusionAD_optimized_profiled.py    # 최적화 후 성능 측정용 벤치마크 코드
├── results/
│   └── benchmark_results.csv          # 성능 측정 결과 (수기 작성용 템플릿)
└── report/
    └── report.pdf                     # 최적화 분석 및 결과 보고서 (추가 예정)
```

설치 방법

```bash
# 가상 환경 생성 및 활성화 (예시)
conda create -n optim_env python=3.10
conda activate optim_env

# 패키지 설치
pip install -r requirements.txt
```

데이터셋 준비 및 경로 설정

본 과제에서 모델 검증에 사용한 원본 데이터셋은 기업의 민감한 산업 데이터이므로 일부만 샘플 데이터만 공유드립니다.
공유된 코드는 해상도(`512x512`)가 일치하는 이미지 포맷이면 정상적으로 동작하므로, 
공개 데이터셋의 이미지를 사용해도 코드를 실행할 수 있습니다.

### 공개 데이터셋을 활용한 벤치마크 실행 가이드

1. 임의의 512x512 공개 데이터셋 준비
   - [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)의 정상/비정상 데이터, 혹은 [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 등 이미지를 다운로드하여 512x512로 리사이징 합니다.
   
2. 폴더 구조 준비
   데이터가 위치할 임의의 로컬 폴더를 다음과 같이 생성하고 이미지들을 넣습니다.
   ```text
   /path/to/public_data/
     ├── normal/      (정상 이미지 모음, ex: image_001.png)
     ├── defect/      (결함 이미지 모음, ex: image_001_var1.png)
     └── synthetic/   (합성 결함 이미지 모음)
   ```
   > 주의: 벤치마크 코드는 정상 이미지와 결함 이미지를 파일명 기반으로 매칭합니다. 결함 이미지 파일명에 `_var1` 과 같은 접미사가 붙어있으면 원본 정상 파일과 매핑되도록 코드가 구성되어 있습니다.

3. 코드 내부 경로 수정
   실행할 파이썬 파일 하단의 `base_config` 또는 `config` 딕셔너리에서 데이터 경로를 방금 생성한 폴더 경로로 변경합니다.
   
   ```python
   # 수정해야 할 코드 하단부 설정 (예: src/after/DiffusionAD_optimized.py)
   base_kwargs = {
       'pretrained_model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
       'normal_image_folder': '/path/to/public_data/normal',         # 여기를 수정
       'defect_image_folder': '/path/to/public_data/defect',         # 여기를 수정
       'defect_synthetic_folder': '/path/to/public_data/synthetic',  # 여기를 수정
       'output_dir': './output',
       ...
   }
   ```
실행 방법

1. 성능 벤치마크 측정

벤치마크 코드는 Batch Size 1과 2에 대한 실행 시간 및 VRAM 사용량을 자동으로 측정하여 화면에 출력합니다.


```bash
# 최적화 전 (Before) 성능 측정
python benchmark/DiffusionAD_baseline_profiled.py

# 최적화 후 (After) 성능 측정
python benchmark/DiffusionAD_optimized_profiled.py
```

