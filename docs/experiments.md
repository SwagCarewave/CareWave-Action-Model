# CSI Stream Experiment Log

## Experiment 1 — Subject Split Baseline

- Split: `yena=train`, `sujin=val`, `hoyeon=test`
- 결과: 새로운 피험자에 대한 일반화 성능 부족
- 상태: 보관용 baseline

## Experiment 2 — Session Split Baseline

- Split: 세 사람의 일반 촬영 파일을 train/validation으로 분리
- Test: 별도 test 폴더의 촬영 4개
- Augmentation: 비활성화
- Validation macro-F1: 0.7941
- Test macro-F1: 0.3746
- Fall precision: 0.0690
- Fall recall: 0.1250
- Fall F1: 0.0889
- 상태: augmentation 비교 기준

## Experiment 3 — Window Centering

- Window centering: 활성화
- 모델 축소 및 regularization 강화
- Validation macro-F1: 0.5542
- 결과: 정적 자세 정보가 제거돼 standing/lying 성능 하락
- 상태: 폐기

## Experiment 4 — Session Split + Augmentation

- Window centering: 비활성화
- Model: 기존 128-channel 구조
- Augmentation: 활성화
- Best epoch: 15
- Validation macro-F1: 0.8501
- Test macro-F1: 0.4320
- Fall precision: 0.2059
- Fall recall: 0.4375
- Fall F1: 0.2800
- Event recall: 0.5
- Mean detection latency: 2.5 seconds
- Known issue: `hoyeon_test_04`의 standing 27개를 모두 오분류
- 상태: 현재 CSI-only 후보
