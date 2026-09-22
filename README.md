# CareWave-Action-Model
# CareWave Action Model

2차 행동 분류 모델을 위한 저장소입니다. 현재 CSI Stream은 3대 RX의 CSI를
10fps, 프레임당 156차원으로 정렬하고 3초(30프레임) 윈도우에서
`standing`, `falling`, `lying`을 분류합니다.

## 데이터 배치

원본과 라벨은 같은 사람 폴더, 같은 sample id를 사용합니다.

```text
data/
├─ raw_csi/
│  ├─ yena/yena_fall_normal_01_csi_raw.csv
│  ├─ sujin/...
│  └─ hoyeon/...
└─ labels/
   ├─ yena/yena_fall_normal_01_labels.csv
   ├─ sujin/...
   └─ hoyeon/...
```

라벨 CSV 열은 `video_name,start_sec,end_sec,label`입니다. 학습 클래스는
`standing`, `falling`, `lying`입니다. 그 밖의 `transition`, `getting_up`,
`ignore`, 걷기·자세 조정 등의 보조 라벨이 윈도우 중앙에 위치하면 해당
윈도우를 제외합니다.

## 실행 순서

```bash
python -m venv .venv
# Windows Git Bash
source .venv/Scripts/activate
pip install -r requirements.txt

python data/build_action_labels.py
python data/build_action_windows.py --config configs/action_fusion.yaml
python data/validate_splits.py
python train_action_classifier.py --config configs/action_fusion.yaml
python evaluate_action_classifier.py --split test
```

한 개의 raw CSV를 학습된 모델로 예측하려면:

```bash
python infer_action_stream.py data/raw_csi/yena/yena_fall_normal_01_csi_raw.csv
```

생성 데이터와 모델 가중치는 각각 `data/processed/csi_stream/`과
`outputs/csi_stream/`에 저장되며 Git에는 올라가지 않습니다.

## 전처리상 주의점

기존 수집 과정에서 amplitude가 0인 원소가 삭제되어, 중간 서브캐리어가
삭제된 행은 원래 인덱스를 복원할 수 없습니다. 현재 코드는 1차 자세 모델과
호환되도록 52개 열을 기준으로 RX별 0.1초 평균과 시간축 보간을 수행하고,
보간 전 결측률을 품질 보고서에 남깁니다. 이후 새 데이터를 수집할 때는
0값을 삭제하지 말고 원래 인덱스에 보존해야 합니다.

## Current CSI Stream Baseline

- Checkpoint: `outputs/csi_stream_augmented/best.pt`
- Validation macro-F1: 0.8501
- Test macro-F1: 0.4320
- Fall precision: 0.2059
- Fall recall: 0.4375
- Fall F1: 0.2800
- Event recall: 0.5
- Known issue: `hoyeon_test_04`의 standing 윈도우 27개를 모두 오분류함
- Note: 현재 test set은 모델 비교에 사용되어 최종 평가 세트가 아닌 개발용 세트로 간주함