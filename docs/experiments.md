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
- 상태: 이전 CSI-only 후보 (Experiment 5로 대체)

## Experiment 5 — Packet Stream Split + Train-only Aux Negatives

- Config: `configs/action_stream_split_v2.yaml` (walk_v1과 같은 split, 같은 val/test 윈도우)
- 발견 1: 각 RX에 스펙트럼 모양이 다른 두 패킷 스트림(상관 약 -0.75)이 무작위로
  번갈아 들어옴. 기존 전처리는 0.1초 bin에서 두 스트림을 섞어 평균해, 스트림 전환
  잡음이 동작 신호를 가림. 기존 특징의 프레임 변화량 중앙값은 standing 4.27,
  walking 4.40, falling 4.82로 거의 차이가 없었고, 낙상 onset 기준 ±8초 평균에서도
  변화가 보이지 않음.
- 수정 1: RX별 두 스트림을 인과적으로 분리(초기 60패킷 2-means 후 EMA 추적)하고,
  스트림별 정규화 스펙트럼 52개와 패킷 간 모양 변화량 1개를 사용함(318-D).
  분리 후 동작량 중앙값은 walking 0.152, falling 0.097, standing 0.065, lying 0.061.
- 발견 2: 낙상 녹화에서 lying/getting_up/transition 구간이 모두 제외되어, 낙상 녹화의
  학습 윈도우는 사실상 낙상뿐이었음. 녹화 자체의 특성만으로 라벨을 맞힐 수 있는 구조.
- 수정 2: 해당 구간을 train split에만 비낙상(`aux_negative`)으로 추가함. val/test
  윈도우 정의는 walk_v1과 동일함.
- Scaler: 새 특징의 robust scaler를 train 윈도우로만 새로 학습함.
- 기각: window 중심화(`window_center_shape: true`)는 val F1이 0.721로 낮아서 제외함.
- 선택 기준: validation만 사용함. Test는 최종 보고에만 사용함.

Test 결과(raw window, 비낙상 509 / 낙상 15 윈도우, 낙상 이벤트 2개):

| 모델 | 실행 | Fall P | Fall R | Fall F1 | FP | AUC | 이벤트 |
|---|---|---|---|---|---|---|---|
| walk_v1 (기존) | seed 42 | 0.048 | 0.733 | 0.089 | 220 | 0.727 | 2/2 |
| walk_v1 (기존) | seed 52 | 0.044 | 0.467 | 0.080 | 152 | 0.650 | 1/2 |
| walk_v1 (기존) | seed 62 | 0.029 | 0.467 | 0.055 | 233 | 0.560 | 1/2 |
| stream split만 | seed 42/52/62 | 0.092–0.099 | 0.933 | 0.168–0.178 | 128–138 | 0.864–0.878 | 2/2 |
| stream split + aux | seed 42 (4 threads) | 0.207 | 0.800 | 0.329 | 46 | 0.944 | 2/2 |
| stream split + aux | seed 52 (4 threads) | 0.206 | 0.467 | 0.286 | 27 | 0.522 | 1/2 |
| stream split + aux | seed 62 (4 threads) | 0.364 | 0.800 | 0.500 | 21 | 0.958 | 2/2 |
| stream split + aux | seed 42 (기본 threads, 저장된 `outputs/csi_stream_split_v2/best.pt`) | 0.115 | 0.733 | 0.198 | 85 | 0.911 | 2/2 |

- 평균: 기존 F1 0.075, FP 201.7 → 최종 설정 4회 F1 0.328, FP 44.8
- 모든 최종 설정 실행의 F1이 모든 기존 실행보다 높고, FP는 모두 더 낮음
- Recall은 개선되지 않음(기존 0.467–0.733, 최종 0.467–0.800). seed 52 실행은
  `yena_test_02`의 낙상을 놓침
- 주의: CPU 스레드 수가 달라지면 같은 seed에서도 결과가 달라짐. 실행 간 편차가 크며
  test 낙상은 윈도우 15개, 이벤트 2개뿐이라 수치의 불확실성이 큼
- 남은 오탐의 대부분은 test 긴 녹화 안의 walking 구간
- 상태: 현재 CSI-only 후보

## Experiment 6 — Walking False Alarms and Event-level Post-processing

- 원인 분석: test 걷기 오탐은 6/25 녹화(`sujin_test_03`, `yena_test_02`)에서 나옴.
  걷기 학습 데이터는 모두 6/24와 6/26 녹화임. 표준화된 동작량 중앙값은 test 걷기
  1.28로, 학습 걷기(6/24 1.63, 6/26 2.31)보다 약하고 6/26 학습 낙상(1.36)과 비슷함.
- 시도(기각): 학습 중 윈도우별 동작 세기를 0.4~1.6배로 무작위 조절함. 4 threads,
  seed 42/52/62 기준 test FP 41/30/26(v2: 46/27/21), FN 6/5/8(v2: 3/8/3)로 개선이
  없어 채택하지 않음. Val F1 평균은 v2와 같은 0.982라 val로는 구분할 수 없었음.
- 새 평가: `evaluate_stream_events.py`. 녹화 전체를 0.5초 간격으로 연속 추론하고
  경보 단위로 채점함. 낙상 onset -1초부터 impact +8초 사이의 경보를 검출로,
  나머지 경보를 오경보로 셈.
- 새 후처리: `fall_then_still`(`postprocess_state_machine.detect_fall_alarms`).
  - 2/3 확인된 후보 구간이 6초 이하로 짧고, 이후 4초 안에 2초 평균 동작량이 0.0
    이하(누워서 조용한 수준)로 내려가야 경보를 울림.
  - 동작 기준 0.0은 train split 분포로 정함: lying 75백분위 -0.07, walking
    5백분위 0.58.
- 선택 규칙(test 전에 정함): val에서 15/15 검출인 설정 중 `fall_then_still` 2/3을
  선택함. 선택한 설정은 val 모델 4개 모두에서 15/15 검출, 오경보 0건이었음.
- Config: `configs/action_stream_split_v3.yaml` (v2와 postprocess만 다름)

Test 경보 결과(녹화 4개, 약 6.2분, 낙상 이벤트 2개):

| 모델 | 후처리 | 검출 | 미탐 | 오경보 | 오경보 중 걷기 | 평균 지연 |
|---|---|---|---|---|---|---|
| walk_v1 저장 모델 | 기존 (EMA, 0.65 이상, 3/5) | 1 | 1 | 13 | 7 | 4.0초 |
| v2 저장 모델 | 기존 | 2 | 0 | 7 | 7 | 2.25초 |
| v2 저장 모델 | fall_then_still | 2 | 0 | 2 | 0 | 4.75초 |
| v2 seed 42 (4 threads) | fall_then_still | 2 | 0 | 2 | 0 | 4.75초 |
| v2 seed 52 (4 threads) | fall_then_still | 1 | 1 | 0 | 0 | 5.0초 |
| v2 seed 62 (4 threads) | fall_then_still | 2 | 0 | 1 | 0 | 4.75초 |

- Val 경보 결과: walk_v1 + 기존 후처리는 8/15 검출, 오경보 2건.
  v2 + fall_then_still은 15/15 검출, 오경보 0건, 평균 지연 3.87초.
- v2 저장 모델에 남은 오경보 2건(`yena_test_02`):
  - 11.0초: 4~8초에 걷다가 멈춰 선 직후
  - 37.0초: `transition` 라벨 구간
- 한계: 짧게 걷고 멈추는 동작은 낙상과 구분하지 못할 수 있음. 경보 지연이
  기존 후처리보다 약 2.5초 늘어남. seed 52 모델의 `yena_test_02` 미탐은
  후처리로 해결되지 않음(모델 확률 자체가 낮음).

## Experiment 7 — Label Audit, Packet Source Column, Seed Ensemble

선택 방법: val이 거의 만점이라(경보 15/15, 오경보 0) 구분력이 없음. 그래서 val이
나빠지지 않는지 먼저 확인하고, test 결과로 채택 여부를 정함. 이번 결정에 test를
사용했으므로 test는 개발용 세트임.

### 4. 낙상 라벨 점검 — 도구 유지, 학습 변경은 되돌림

- `data/audit_fall_events.py`: 이벤트별 CSI 동작 최고점과 라벨 시각의 차이를
  `data/metadata/fall_event_audit.csv`에 기록함. `review_status`와 라벨은 바꾸지 않음.
  - 119건 중 65건 표시: near_recording_start 45, peak_far_from_label 17, weak_motion 7
  - 최고점 시각의 중앙값은 onset +1.0초
  - 의심 이벤트(시각 어긋남 또는 약한 신호) 24건: train 21, val 3, test 0
  - 대부분 최고점이 라벨보다 약 2초 앞섬. 여러 번 넘어지는 녹화에서 직전의 일어서는
    동작일 수 있어 라벨 오류로 단정할 수 없음. 영상 검토가 필요함.
- 시험: train의 의심 이벤트 21건을 학습에서 제외함(train 낙상 윈도우 576 → 456).

| seed 42/52/62 (4 threads) | v2 | 의심 이벤트 제외 |
|---|---|---|
| Val F1 | 0.973 / 1.000 / 0.974 | 0.960 / 0.987 / 0.987 |
| Test FP | 46 / 27 / 21 | 44 / 71 / 47 |
| Test FN | 3 / 8 / 3 | 3 / 4 / 2 |
| 앙상블 val FN / test FP / test FN | 1 / 21 / 4 | 2 / 44 / 4 |

- 결과: 오탐이 늘고 앙상블 기준으로도 나빠서 되돌림. 점검 도구만 남김.

### 5. 패킷 송신원 열 지원 — 유지 (기존 데이터 결과 변화 없음)

- `stream_split.source_column`(예: `source_mac`): raw CSV에 이 열이 있으면 모양
  추정 대신 이 값으로 두 스트림을 나눔.
  - 추정으로 얻은 스트림 번호를 이 열로 넣은 합성 CSV에서, 두 방식의 특징이
    완전히 같게 나오는 것을 확인함.
- 품질 보고서에 RX별 `stream_assignment`와 `stream_separation`을 추가함.
  - `stream_separation` 전체 중앙값 0.20
  - 가장 낮은 녹화: `yena_fall_normal_01` RX3 0.013, `sujin_fall_stay_down_04` RX2 0.040
- 기존 v2 전처리 결과(X, y, 가중치)가 바이트 단위로 같음을 확인함.
- 앞으로 수집할 때 송신원 MAC 같은 패킷 출처를 CSV에 함께 기록하길 권장함.

### 6. Seed 앙상블 — 채택

- `make_ensemble.py`: 여러 seed 체크포인트의 낙상 확률을 평균하는 체크포인트
  하나를 만듦. 임계값은 단일 모델과 같은 규칙으로 val에서 다시 고름(0.55).
- 구성원: v2 설정의 seed 42/52/62 모델(4 threads로 학습)
- 저장: `outputs/csi_stream_split_v2_ensemble/best.pt`
- 평가, 추론 스크립트는 `models/loading.py`로 앙상블과 단일 체크포인트를 모두 읽음

| | v2 저장 모델 (단일) | 3-seed 앙상블 |
|---|---|---|
| Val window FP / FN | 0 / 1 | 0 / 1 |
| Test window FP / FN (509 / 15) | 85 / 4 | 21 / 4 |
| Test fall P / R / F1 | 0.115 / 0.733 / 0.198 | 0.344 / 0.733 / 0.468 |
| Test AUC | 0.911 | 0.954 |
| Val 경보: 검출 / 오경보 | 15/15, 0 | 15/15, 0 |
| Test 경보: 검출 / 오경보 / 평균 지연 | 2/2, 2, 4.75초 | 2/2, 1 (transition), 4.75초 |

- 남은 test 오경보 1건: `yena_test_02` 37.0초, `transition` 라벨 구간
- Window 미탐 4개는 그대로임(`yena_test_02` 낙상 앞쪽 윈도우)
- 상태: 현재 CSI-only 후보

## Experiment 8 — 새 test 녹화 2개로 고정 평가 (모델/설정 변경 없음)

- 추가 파일: `csi_test_01`(2026-06-25), `yena_test_03`(2026-06-26)
  - `csi_raw_test01.csv`는 이름 규칙에 맞게 `csi_test_01_csi_raw.csv`로 바꿈
  - 새 라벨 `bending_over`, `straightening_up`을 허용 라벨과 hard negative(비낙상)에 추가함
  - `csi_test_01` 낙상 2건을 `action_events.csv`에 추가함(needs_review, 기존 행 변경 없음)
- 검증: train/val 윈도우, 기존 test 4개 윈도우, scaler가 이전과 바이트 단위로 같음
- 평가 모델: `outputs/csi_stream_split_v2_ensemble/best.pt`와 v3 후처리를 그대로 씀.
  결과를 보고 모델, 임계값, 후처리를 바꾸지 않음.
- 두 녹화는 모델이 처음 보는 녹화지만, 학습 데이터와 같은 날짜(6/25, 6/26)에 찍은 것임.

새 녹화 2개 결과 (비낙상 400 / 낙상 14 윈도우, 낙상 이벤트 2개):

| | 처음 모델 (walk_v1) | 현재 앙상블 |
|---|---|---|
| Window FP | 254 / 400 (63.5%) | 69 / 400 (17.2%) |
| Window FN | 13 / 14 | 13 / 14 |
| 낙상 윈도우 탐지율 | 7.1% (1/14) | 7.1% (1/14) |
| 경보 검출 | 2 / 2 (지연 6.0초, 10.0초) | **0 / 2** |
| 오경보 | 4 | 1 (`yena_test_03` 72.5초 걷기) |

- 처음 모델의 경보 2/2는 오탐률 63.5% 상태에서 나온 것임. `yena_test_03`은
  윈도우 212개 모두 낙상으로 판정함.

현재 앙상블의 오류 위치:

- 오탐 69개
  - `yena_test_03`: 걷기 60, 허리 펴기 1
  - `csi_test_01`: 걷기 8
  - 허리 숙이기와 서 있기는 0개
- `yena_test_03` 걷기 오탐은 걷기 구간 중간에 많음(가장자리 10/34, 중간 50/73).
  이 녹화의 걷기 동작량은 2.16으로 학습 걷기(2.31)와 비슷해서, "약한 걷기" 가설로
  설명되지 않음.
- 미탐
  - `csi_test_01` 낙상 1(94~96초): 걷다가 넘어진 뒤 누워 있지 않고 바로 일어남.
    CSI 동작량은 2.3~2.5로 뚜렷하지만 앙상블 확률은 최고 0.60이고 곧 떨어짐.
    새 후처리는 낙상 뒤 멈춤을 요구해서 경보가 울리지 않음.
  - `csi_test_01` 낙상 2(104~107초): 걷다가 넘어지고 2초만 누워 있음.
    동작량 0.9~1.3, 앙상블 확률 약 0.3으로 임계값 0.55 미만.
  - 학습 낙상 녹화는 대부분 "서 있다가 넘어짐 → 5초 이상 누워 있음" 형태여서
    "걷다가 넘어짐", "넘어지자마자 일어남"을 거의 보지 못했을 가능성이 큼.

결론: 새 녹화에서 오탐은 처음 모델보다 적지만, 낙상 검출은 실패함(0/2).
기존 test 4개에서 본 개선이 새 녹화에서 재현되지 않았음.

## Experiment 9 — 기존 데이터로 개선 시도 (모두 되돌림)

평가 기준(시작 전에 정함): val 경보 15/15 유지와 val window F1 비하락이 전제 조건.
그다음 test 6개 합계의 낙상 검출(많을수록), 오경보(적을수록), window FP+FN(적을수록)
순으로 비교함. test 6개는 이미 결과를 본 개발용 세트임.

기준(현재 앙상블):
- val: FP 0, FN 1, 경보 15/15, 오경보 0
- test 6개: FP 90, FN 17, 경보 2/4, 오경보 2

### A. 낙상 가중치 수정 — 되돌림

- 발견: 모든 낙상 이벤트의 `label_confidence`가 검토 전 임시값 0.5임.
  이 값이 학습 가중치에 곱해져 낙상 윈도우 가중치가 0.5가 됨.
  hard negative는 4.0이라 낙상은 전체 loss 가중치의 3.4%임.
- 시도: 임시값을 쓰지 않도록 해서 낙상 가중치를 1.0으로 올림. 3 seed 앙상블로 평가함.
- 결과
  - val은 같음(FP 0, FN 1, 15/15)
  - test 6개: FP 144, FN 18, 경보 2/4, 오경보 3
  - 새 녹화 낙상 윈도우 탐지 0/14
- 오탐만 늘고 놓친 낙상은 여전히 못 잡아서 코드 변경을 되돌림.

### B. 윈도우 내 스펙트럼 중심화(위치 정보 제거) — 중단

- 가설: 걷기 오탐이 학습 낙상 장소 근처를 지날 때 생김.
- 결과: 5 epoch까지 val F1이 0.40~0.62에 머묾(train F1 0.98).
  기존 설정은 4 epoch에 0.97에 도달했음. val 조건에 명확히 걸려서 학습을 중단함.
  이 설정은 config 옵션으로만 존재하고 기본값은 꺼져 있음.

결론: 현재 데이터로 시도한 두 개선이 모두 성능을 낮춤. 현재 모델을 유지함.
놓친 낙상 형태(걷다가 넘어짐, 바로 일어남)의 학습 데이터가 필요함.
