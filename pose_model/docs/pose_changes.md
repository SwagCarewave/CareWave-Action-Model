# Pose 파트 변경점

설계서와 팀 폴더 구조 안내에서 바뀐 부분만 적었습니다.

## 설계서·폴더 구조 대비

| 항목 | 원래 | 바뀐 내용 | 이유 |
|---|---|---|---|
| 분류 | 서기·눕기·낙상 3개 | 서기(0)·낙상(1) 2개, 출력 1개, BCE 손실 | 설계서 최종안 |
| 입력 좌표 | 정답(MediaPipe) 가능 | 1차 모델이 예측한 좌표만 사용 | 설계서 4.1 |
| 좌표 주기 | 언급 없음 | 10Hz (1차 모델을 1프레임씩 밀면서 추론) | 낙상 전환 구간을 촘촘히 보기 위해 |
| 기준 모델 | CSI-only, Pose-only | Pose-only만 | Pose 파트 범위 |
| 코드 위치 | `features/`, `models/` 등 팀 폴더 | 전부 `pose_model/` 안 | 팀원이 같은 파일을 수정 중이라 겹치지 않게 |
| 학습·평가 파일 | `train_action_classifier.py` 등 | `pose_model/train_pose_classifier.py`, `evaluate_pose_classifier.py` | 위와 같음 |
| 설정 | `configs/action_fusion.yaml` | `pose_model/configs/pose_branch.yaml` | 위와 같음 |

## 라벨
- `falling` 구간의 앞 0.5초 ~ 뒤 1.0초를 낙상(1)으로 봄. (onset, impact 표시가 따로 없어서 `falling` 구간으로 대신함)
- 서 있음, 낙상 뒤 누워 있음, 일어나는 중, 걷기, 앉기, 줍기 등은 낙상 아님(0)
- `yena_fall_normal_03`은 라벨은 있으나 1차 모델 목록에 없어 제외

## 설계서와 다르게 한 것
- 학습 중 조기 종료 기준: 검증 AUPRC (설계서는 검증 이벤트 F1. 검증 이벤트가 너무 적어서)
- 데이터 변형: 좌표 대신 특징에 잡음, 시간 이동(±1프레임), 관절 가리기
- 알림 판단의 기준값·평활·연속 조건은 검증에서만 정하고 TEST는 한 번만 평가
- 만들지 않은 것: unknown 판단 규칙, transition score

## 알게 된 것
- 1차 모델이 학습에 쓴 영상은 좌표가 정답에 가까워 점수가 부풀려짐. 1차 모델 복사본을 5조각으로 다시 학습해서 모든 영상을 처음 보는 상태로 만들어 평가함
- 1차 모델 스케일러 파일 위치는 설계서와 다름: `model_ready_315/csi_robust_scaler_train_only.npz`
- `features_315/train`에는 1차 학습 녹화와 검증 녹화가 함께 들어 있음
- carewaveTest 모델은 y축 방향이 반대라서 뒤집어서 사용함
- 결과는 `docs/pose_report.ipynb` 참고
