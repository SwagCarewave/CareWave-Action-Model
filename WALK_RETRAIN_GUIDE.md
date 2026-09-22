# 걷기 데이터 추가 재학습 — walk_v1

압축을 풀고 csi_dataset.py와 configs/action_walk_v1.yaml을 기존 CareWave-Action-Model 폴더의 같은 위치에 복사합니다. csi_dataset.py는 먼저 백업해 두세요. 원본 raw CSV와 라벨은 변경하지 않습니다.

## 변경 사항
- csi_dataset.py: 헤더가 sub_50까지만 있고 행별 CSI 값이 51/52개인 기존 수집 파일을 읽도록 수정했습니다. 행을 건너뛰지 않고 부족한 끝 값을 결측값으로 채워 기존 보간에 전달합니다. 원래 삭제된 서브캐리어 위치는 복원할 수 없습니다.
- configs/action_walk_v1.yaml: 기존 학습 설정을 유지하고 전처리 결과 경로만 data/processed/csi_stream_walk_v1로 분리했습니다.
- 모델 구조, hard_negative_weight=4.0, 낙상 기준, 후처리 설정은 변경하지 않았습니다. 이번 실행은 걷기 데이터 추가 효과를 확인하는 비교 실험입니다.
- walking은 기존 코드에서 비낙상(0)으로 들어갑니다. 출력 이름 standing은 실제로 걷기를 포함한 비낙상을 뜻합니다.

## 확인 결과
104개 raw/label 쌍, 걷기 28개, 라벨 형식 오류 없음.
걷기 학습: 21개 영상 / 1028개 윈도우.
걷기 검증: 7개 영상 / 460개 윈도우.
전체: X=(3699,30,315).
기존 test: 524개 윈도우(비낙상 509, 낙상 15). 이전 test 입력, 정답, 가중치와 동일함을 확인했습니다.
각 촬영 파일은 한 split에만 속합니다. 동일 긴 촬영을 여러 파일로 나눈 경우의 누수 여부는 파일명만으로 확인할 수 없습니다.

## 실행 전 알아둘 점
낙상 이벤트 119개가 needs_review 상태입니다. 아래 --allow-unreviewed-events는 기존 이벤트 시점을 그대로 사용하는 임시 실험 옵션입니다. 검토 완료로 바꾸지 않습니다. onset/impact 시점이 검토된 실험으로 보고하면 안 됩니다. 이벤트를 실제로 검토한 뒤 reviewed로 기록했다면 이 옵션을 빼세요.
yena_walk_normal_01 라벨 끝은 124초, raw 길이는 약 122.88초입니다. raw 끝 이후 입력을 새로 만들지 않습니다. 영상과 raw의 시작 시점이 같다는 가정은 별도로 확인해야 하며, 임의로 라벨을 당기지 않았습니다.
기존 윈도우 코드는 비낙상 비율이 80% 이상이면 채택하므로 박수 등 제외 구간이 일부 섞인 경계 윈도우가 남을 수 있습니다. 이번 비교에서는 기존 규칙을 유지했습니다.

## 실행 명령
기존 Python 학습 환경을 활성화하고 프로젝트 루트에서 순서대로 실행합니다. 오류가 발생하면 다음 명령으로 넘어가지 마세요.

```bash
python data/build_action_windows.py --config configs/action_walk_v1.yaml --allow-unreviewed-events
python data/validate_splits.py --config configs/action_walk_v1.yaml
python train_action_classifier.py --config configs/action_walk_v1.yaml --output-dir outputs/csi_stream_walk_v1 --seed 42
python evaluate_action_classifier.py --config configs/action_walk_v1.yaml --checkpoint outputs/csi_stream_walk_v1/best.pt --split val --output outputs/csi_stream_walk_v1/val_evaluation.json
python evaluate_action_classifier.py --config configs/action_walk_v1.yaml --checkpoint outputs/csi_stream_walk_v1/best.pt --split test --output outputs/csi_stream_walk_v1/test_evaluation.json
```

기존 build_action_windows.py가 직접 raw/label 쌍과 분할을 다시 생성하므로 이번 실행에 build_action_labels.py를 다시 실행할 필요는 없습니다.
기존 모델 경로와 전처리 경로는 덮어쓰지 않습니다. 같은 walk_v1 명령을 재실행하면 walk_v1 결과는 덮어씁니다.

## 결과 확인
outputs/csi_stream_walk_v1/history.csv와 test_evaluation.json을 확인합니다.
raw_window_metrics.confusion_matrix는 [[TN,FP],[FN,TP]] 순서입니다. FP 감소와 낙상 recall을 함께 비교하세요. 후처리 결과는 postprocessed_window_metrics에 별도로 있습니다. 이것은 윈도우 평가이며 실시간 이벤트 검출률이나 경보 지연 평가를 대신하지 않습니다.
모델과 임계값은 validation으로 선택합니다. 이미 여러 번 비교한 test는 개발용 평가로 해석하고, test 결과에 맞춰 임계값을 반복 조정하지 마세요.
hard negative가 크게 늘었지만 가중치 4.0은 유지했습니다. 미탐이 늘면 validation 기준으로 가중치를 낮춘 실험과 비교할 수 있습니다. 이번 결과를 보기 전에 개선을 확정할 수는 없습니다.

## 수행한 검증
전체 raw 104개 전처리와 split 검증을 완료했습니다. 가변 길이 CSV의 메타데이터 정렬, 누락값 처리, 과도한 열 거부, 정상 CSV 동작 유지를 확인했습니다. 이 환경에는 PyTorch가 없어 학습과 평가 추론은 실행하지 않았습니다.
