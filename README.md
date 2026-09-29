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

## Current CSI Stream Candidate — stream split v2

```bash
python data/build_action_windows.py --config configs/action_stream_split_v2.yaml --allow-unreviewed-events
python data/validate_splits.py --config configs/action_stream_split_v2.yaml
python train_action_classifier.py --config configs/action_stream_split_v2.yaml --output-dir outputs/csi_stream_split_v2 --seed 42
python evaluate_action_classifier.py --config configs/action_stream_split_v2.yaml --checkpoint outputs/csi_stream_split_v2/best.pt --split test --output outputs/csi_stream_split_v2/test_evaluation.json
python infer_action_stream.py data/raw_csi/test/yena_test_02_csi_raw.csv --config configs/action_stream_split_v2.yaml --checkpoint outputs/csi_stream_split_v2/best.pt
```

- 각 RX의 두 패킷 스트림을 분리해 318-D 특징을 만듦(`feature_mode: stream_split`)
- Build 시 train 윈도우로 scaler를 새로 학습함:
  `data/preprocessing/csi_stream_split_scaler_train_only.npz`
- 저장된 체크포인트의 test 결과: Fall P 0.115, R 0.733, F1 0.198, FP 85/509, AUC 0.911
- 같은 설정 4회 학습의 test F1 평균은 0.328(범위 0.198–0.500)
- 비교: walk_v1 기존 모델 3회 평균 F1 0.075, FP 201.7
- 상세 결과와 주의점: `docs/experiments.md` Experiment 5

경보 단위 평가(녹화 전체 연속 추론 + `fall_then_still` 후처리):

```bash
python evaluate_stream_events.py --config configs/action_stream_split_v3.yaml --checkpoint outputs/csi_stream_split_v2/best.pt --split test --output outputs/csi_stream_split_v2/stream_events_test.json
```

- Test 결과: 낙상 2/2 검출, 오경보 2건(걷기 0건), 평균 지연 4.75초
- 비교: walk_v1 + 기존 후처리는 1/2 검출, 오경보 13건
- 상세 결과: `docs/experiments.md` Experiment 6

### 현재 후보: 3-seed 앙상블

```bash
# 구성원 학습 (앙상블 구성원은 OMP_NUM_THREADS=4로 학습함; 스레드 수가 다르면 결과가 조금 달라짐)
for s in 42 52 62; do OMP_NUM_THREADS=4 python train_action_classifier.py --config configs/action_stream_split_v2.yaml --output-dir outputs/csi_stream_split_v2_seeds/seed_$s --seed $s; done
python make_ensemble.py --config configs/action_stream_split_v3.yaml --members outputs/csi_stream_split_v2_seeds/seed_42/best.pt outputs/csi_stream_split_v2_seeds/seed_52/best.pt outputs/csi_stream_split_v2_seeds/seed_62/best.pt --output outputs/csi_stream_split_v2_ensemble/best.pt
python evaluate_stream_events.py --config configs/action_stream_split_v3.yaml --checkpoint outputs/csi_stream_split_v2_ensemble/best.pt --split test --output outputs/csi_stream_split_v2_ensemble/stream_events_test.json
```

- Test window: FP 21/509, FN 4/15, fall F1 0.468, AUC 0.954
- Test 경보: 낙상 2/2 검출, 오경보 1건, 평균 지연 4.75초
- 낙상 라벨 점검 목록: `python data/audit_fall_events.py` → `data/metadata/fall_event_audit.csv`
- 상세 결과: `docs/experiments.md` Experiment 7

## Previous CSI Stream Baseline

- Checkpoint: `outputs/csi_stream_augmented/best.pt`
- Validation macro-F1: 0.8501
- Test macro-F1: 0.4320
- Fall precision: 0.2059
- Fall recall: 0.4375
- Fall F1: 0.2800
- Event recall: 0.5
- Known issue: `hoyeon_test_04`의 standing 윈도우 27개를 모두 오분류함
- Note: 현재 test set은 모델 비교에 사용되어 최종 평가 세트가 아닌 개발용 세트로 간주함