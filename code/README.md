# 웨이퍼 결함 분류 — 멤리스터 PIM 가속기

VTEAM 멤리스터 기반 PIM 가속기를 활용한 웨이퍼 맵 결함 분류 시스템

**최종 성능: F1 = 0.9881 (98.8점)**

---

## 실행 순서

```bash
cd base
python step1_preprocessing.py   # 1. 데이터 전처리
python step2_train.py            # 2. 순수 CNN 학습
python step3_evaluate.py         # 3. 멤리스터 매핑 후 평가
```

---

## 폴더 구조

```
capstone_design/
├── base/                       ← 기본 코드
│   ├── model.py                   공통 모듈 (CNN, Dataset, FocalLoss)
│   ├── step1_preprocessing.py     데이터 전처리 (리사이징 + 분리)
│   ├── step2_train.py             Ex-Situ 학습 (순수 CNN, 30 epoch)
│   └── step3_evaluate.py          Ex-Situ 평가 (멤리스터 매핑)
│
├── optimize.py                 ← v1 파라미터 탐색 (73조합 자동 실험)
├── optimize_v2.py              ← v2 성능 극대화 (Noise-Aware + threshold)
│
├── reports/                    ← 보고서
│   ├── final_report (.md/.html)   최종 보고서
│   ├── easy_report (.md/.html)    쉬운 설명서
│   └── paper (.md/.html)          논문 형식
│
├── data/                       ← 전처리된 데이터 (gitignore)
├── results/                    ← 실험 결과 (gitignore)
└── README.md
```

---

## 파일별 설명

### `base/` — 기본 코드

| 파일 | 역할 | 설명 |
|---|---|---|
| `model.py` | 공통 모듈 | 3층 CNN 모델, 데이터셋 클래스, Focal Loss 정의. step2, step3에서 import하여 사용 |
| `step1_preprocessing.py` | 전처리 | 원본 데이터(52×52) → 64×64 리사이징, train/test 분리 후 학습 데이터만 밸런싱 |
| `step2_train.py` | 학습 | 멤리스터 없이 순수 CNN만 30 epoch 학습. 가중치를 `software_model.pth`로 저장 |
| `step3_evaluate.py` | 평가 | 저장된 가중치를 멤리스터에 매핑하고 성능 측정. 소자(TiO2/HfO2/TaO2) 변경 가능 |

### 최적화 스크립트

| 파일 | 역할 | 설명 |
|---|---|---|
| `optimize.py` | v1 탐색 | 73가지 멤리스터 파라미터 조합 자동 실험 (tile, ADC, r_on/r_off, 변동성 등) |
| `optimize_v2.py` | v2 극대화 | epoch 50 재학습 + Noise-Aware Training + threshold 튜닝 (0.3~0.7) |

### `reports/` — 보고서

| 파일 | 대상 | 설명 |
|---|---|---|
| `final_report` | 일반 | 전체 프로젝트 과정, 결과, 코드 구조 종합 |
| `easy_report` | 입문자 | 멤리스터가 뭔지부터 결과까지 쉽게 설명 |
| `paper` | 학술 | 서론/방법/결과/고찰/결론/참고문헌 논문 형식 |

---

## 소자 변경 방법

`base/step3_evaluate.py`의 설정 섹션에서 변경:

```python
selected_material = "TaO2"   # "TiO2", "HfO2", "TaO2"
TILE_SHAPE = (128, 128)
ADC_RES = 8                  # None = 양자화 없음
```

---

## 환경

- Python 3.11 / PyTorch 2.11.0 / memtorch 1.1.6 / scikit-learn 1.8.0
