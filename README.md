# capstone_data — SO-ARM101 잔차 기반 외력 추정 (실험 데이터 + 스크립트)

저가형 직렬 버스 서보(STS3215) 기반 SO-ARM101 5-DOF 팔에서, **토크 센서 없이** 내장 채널(load/current)의
상태-예측 잔차를 외력 proxy로 쓰는 채널 선별 + 융합 파이프라인의 **재현용 데이터/코드 종합본**이다.



---

## 1. 디렉터리 구조

```
capstone_data/
├── README.md
├── code/                         # 모델 산출 스크립트 + 산출물(JSON) + 결과 로그(txt)
│   ├── featlib.py                # 채널 로더 + 후보 특징 라이브러리(LOAD 44개)
│   ├── search.py                 # GramCV: regime-CV(K_FOLDS=8) ridge subset 평가 하네스
│   ├── ... (아래 2절 참조)
│   └── results/*.txt             # 각 단계 실행 출력 로그
└── data/excitation/              # 실제 사용된 궤적 CSV 14개 (~207 MB)
    ├── run_1 .. run_5/excitation_recording.csv          # 풀(pool): 모델 적합·선택용 (무부하)
    ├── test_noload/                                     # held-out #1 (무부하 여진)
    ├── test_noload_2 .. _5/excitation_recording.csv     # held-out #2~#5 (다른 시드)
    ├── test_noload_p2p/excitation_recording.csv         # held-out #6 (다른 유형: point-to-point)
    └── payload_{84,146,227}g/run_1/excitation_recording.csv   # 외력 검출 평가용 (질량 부착)
```

> 주의: 선택(selection)에는 `run_1..5`만 사용한다. `test_*` 6개는 일반화 검증 전용으로 **선택에 일절 쓰지 않는다.**
> payload 데이터는 외력(추가 질량) 검출 SNR 평가에만 쓴다.

---

## 2. 데이터 형식 (`excitation_recording.csv`)

- 수집 주기 **3 ms** (≈ 333.3 Hz). 한 무부하 여진 궤적 = 120,173 샘플 ≈ 360.5 s. p2p = 74,502 샘플 ≈ 3.7 분.
- 컬럼: `t_ms, step_idx, q1_des..q5_des, q1_meas..q6_meas, spd1_fw..spd6_fw, load1..load6, cur1_raw..cur6_raw, voltage, valid`
- 본 분석은 관절 J2/J3/J4(중력 영향이 큰 관절)에 집중한다.
- `load*` = 부호 있는 부하율(토크 합에 비례), `cur*_raw` = 부호 없는 전류 크기(정류된 토크 크기에 비례). 둘 다 **비교정(uncalibrated)**.
- 속도·가속도는 후처리 Savitzky–Golay 미분으로 얻는다(상수는 `constants6.json` 등 참조).

---

## 3. 스크립트 맵 (실행 순서 ≈ 파이프라인)

LOAD 채널 (state→load 모델):

| 파일 | 역할 | 산출물 |
|---|---|---|
| `featlib.py` | 채널 로더 + 후보 특징 라이브러리(44개) | (모듈) |
| `search.py` | regime-CV(K_FOLDS=8) ridge subset 평가 하네스 | (모듈) |
| `unified.py`, `final.py` | 1차 forward selection + 최종 모델 | `selected_features.json`, `final_model.json` |
| `sel5.py`, `final5.py`, `const5.py` | 5-test 재선택/상수 | `selected5.json`, `final_model5.json`, `constants5.json` |
| `discovery.py`, `stage2_select.py`, `stage2b_augment.py` | x0 어휘 밖 잔차 탐색 + 확장 라이브러리 재선택 | `selected_ext.json` |
| `const6.py`, `final6.py` | 6-test 상수 재정당화 + 최종 LOAD 모델 | `constants6.json`, `final_model6.json` |
| `physics_check.py` | 사전등록 물리 예측(부호·구조·순서) 검증 | `physics_prior.json`(동결), `physics_check.json` |
| `aL_sign_check.py` | 관성계수 음수의 물리적 필연성 검증 | `results/aL_sign_check.txt` |

CURRENT 채널 (보조):

| 파일 | 역할 | 산출물 |
|---|---|---|
| `cur_probe.py` | CURRENT 원신호 프로브 + baseline R²(21특징) | `results/cur_probe.txt` |
| `cur_search.py` | 크기 라이브러리(15개) 개방형 forward selection | `cur_selected.json`, `results/cur_search.txt` |
| `cur_const.py` | CURRENT 상수 재정당화 | `cur_constants.json`, `results/cur_const.txt` |

융합 / 기타:

| 파일 | 역할 | 산출물 |
|---|---|---|
| `fusion.py` | LOAD·CURRENT 잔차 영역분석 + 노이즈정규화 융합 SNR | `fusion_summary.json`, `results/fusion.txt` |
| `gen_pointpoint.py` | held-out용 point-to-point 궤적 생성기 | (궤적) |

### 최종 산출물 요약
- **LOAD**: slim-10 특징, 6-held-out meanTest **R² 0.951** (주 채널). → `final_model6.json`, `selected5.json`, `constants6.json`
- **CURRENT**: slim-6 크기특징, meanTest **R² 0.764** (baseline 0.106 → 7배 개선, 보조 채널). → `cur_selected.json`, `cur_constants.json`
- **융합**: 노이즈정규화 가법, 9/9 케이스에서 단일 최고 채널 대비 검출 SNR 개선. → `fusion_summary.json`

---

## 4. 재현 방법

1. 의존성: Python 3 + `numpy`, `pandas`, `scipy`.
2. **데이터 경로 설정**: `code/featlib.py`의 `EXC` 변수를 이 저장소의 `data/excitation` 절대경로로 바꾼다.
   (현재는 원작성 환경의 절대경로로 하드코딩되어 있음.)
3. `code/` 디렉터리에서 각 스크립트를 실행하면 `code/results/*.txt`와 `*.json`이 재생성된다.
   예) `cd code && python fusion.py`

---

## 5. 한계 (정직한 표기)
- 비교정 채널이라 N·m 절대 단위 추정은 불가. 목표는 **검출 + 상대 크기**.
- 검증된 외력은 정적 중력 부하(추가 질량)뿐.
- 단일 하드웨어 실증. 다기구 일반화는 향후 과제.
