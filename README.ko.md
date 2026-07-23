# hurotics-autolabel

[English](README.md) | **한국어**

착용형 센서(IMU)에서 나온 원본 로그를 읽어서, 녹화된 모든 순간에 대해 착용자가
**서 있는지** 아니면 **걷고 있는지** 판정을하고, 그리고 그 판정에 얼마나 확신하는지
알려줍니다는 파이프라인입니다.

핵심 설계는 이렇습니다. 숫자 계산은 전부 일반 코드가 하고, AI는 오직 "판단"이 필요한 곳에서만
씁니다. 그리고 모든 결정은 기록으로 남아 언제든 다시 재현할 수 있습니다.

이 문서는 프로젝트를 처음 받은 상태에서 최종 결과물까지 가는 방법을 안내합니다!

파이프라인의 대한 세부적인 설명을 읽고 싶으시다면 [`PIPELINE.md`](PIPELINE.md)을
어떠한 과정을 거쳐서 만들었는지를 알고 싶으시다면 [`BUILDLOG.md`](BUILDLOG.md)를 참고하세요
(이 두 문서는 모두 영문으로 되어있는 부분 양해 부탁드립니다ㅠㅠ)

---

## 배경 지식

이 프로젝트에는 성격이 다른 두 종류의 "AI"가 등장하고, 각자 하는 일이 다릅니다:

**머신러닝(ML)은 예시를 보고 패턴을 배우는 프로그램입니다**
여기서는 사람이 미리 "서 있음" 또는 "걷기"라고 표시해 둔 수많은 센서 데이터 구간을 보여 주면, 
스스로 새로운 구간에 라벨을 붙일 수 있게 됩니다

장점: 많은 예시로부터 일반화하는 데 아주 뛰어납니다
단점: 속을 들여다보기 어려운 "블랙박스"입니다- 그래서 틀렸는데도 자신 있게 틀릴 수 있습니다

**에이전트 AI 는 파일을 읽고 그에 대해 추론할 수 있는 언어 모델입니다**
이 친구가 챗봇 뒤에 있는 그 AI입니다 여기서는 고정된 규칙만으로는 내리기 어려운 **판단**에만 사용합니다.
예를 들면 "이 파일은 좀 이상한데, 이미 알려진 문제일까 아니면 처음 보는 문제일까?" 또는 "여기서 물리 
계산이 사람이 붙인 라벨과 일치하는가?" 같은 것입니다

장점: 추론에는 강함
단점: 데이터를 직접 건드리거나 숫자를 계산하는 일은 절대 허용되지 않습니다

**왜 둘 다 쓰고, 왜 이런 구조인가**
이 프로젝트를 관통하는 원칙은 *일은 코드가 하고, AI는 그 일을 판단만 한다* 입니다

- 측정할 수 있는 모든 것 (데이터 정제, 분류기 학습, 성능 채점)은 결정론적 코드와 머신러닝이 처리합니다
- AI 에이전트는 정해진 몇몇 지점에서만 의견을 내고, 스스로는 아무것도 바꿀 수 없습니다

이 구조가 중요한 이유는 시스템 전체를 정직하게 유지해 주기 때문입니다. 모든 숫자는 코드를 다시
돌리면 그대로 재현되고, 모든 AI의 결정은 그 근거와 함께 기록되며, 기계가 정말로 판단할 수 없을
때는 추측하지 않고 "모르겠다"고 말한 뒤 사람에게 넘깁니다. 그리고 작업을 여러 단계(S1 정제 -> 
S2 학습 -> S3 물리 -> S4 결합)로 나눴기 때문에, 각 부분을 따로따로 개선하고 검증할 수 있습니다.

---

## 설치

**Python 3.10 이상**이 필요합니다. 아래 명령어는 모두 프로젝트 폴더 안의 터미널에서 실행합니다.

**1. 프로젝트를 엽니다.**
- VS Code에서 `hurotics-autolabel` 폴더를 엽니다 (File -> Open Folder)
- 그다음 폴더 안에서 터미널을 엽니다 (Terminal -> New Terminal)

**2. 가상 환경을 만들고 켭니다.** (이 프로젝트만을 위한 격리된 Python 공간입니다)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

만약 PowerShell이 실행 스크립트를 막으면, 아래 명령을 한 번 실행한 뒤 다시 시도하세요:
`Set-ExecutionPolicy -Scope Process RemoteSigned`

(macOS나 Linux에서는 켜는 명령이 `source .venv/bin/activate` 입니다)

> **참고:** 새 터미널을 열 때마다 가상 환경을 다시 켜야 합니다. 이걸 깜빡하는 것이 "모듈을 찾을 수
> 없음(module not found)" 오류의 가장 흔한 원인입니다.

**3. 패키지를 설치합니다.**

```powershell
pip install -r requirements.txt
```

**4. API 키를 설정합니다. (AI 에이전트 단계에서만 필요합니다.)** 
이전트는 Claude API를 사용하는데, 이는 사용한 만큼 요금이 붙는 유료 서비스이며 키가 필요합니다.

- Claude 콘솔에서 키를 발급받으세요: **https://platform.claude.com**
  (계정을 만든 뒤 API keys -> Create key).
- 콘솔에서 **월 지출 한도(spend cap)를 꼭 설정하세요.** 비용이 예상 밖으로 새어 나가지 않도록
  막아 주는 안전장치이며, 프로젝트 내부에서는 걸 수 없습니다.
- 예시 설정 파일을 복사한 뒤, 그 안에 키를 붙여넣습니다:

```powershell
Copy-Item .env.example .env
```

  그다음 `.env` 파일을 열어 `ANTHROPIC_API_KEY=` 뒤에 키를 넣으면 됩니다.

> **비용 안내:** 파이프라인의 결정론적 부분 전체(정제, 학습, 채점, 최종 리포트)는 **무료**로
> 돌아갑니다. 키도, 요금도 필요 없습니다. 오직 네 개의 AI 에이전트 단계만 요금이 들며, 한 번 다
> 돌려도 대략 몇 달러 수준입니다. 에이전트 단계를 빼면 키 없이 모든 것을 할 수 있습니다.

---

## 데이터 넣기

두 종류의 데이터가 각각 다른 곳으로 들어갑니다. 파일은 파일 탐색기나 VS Code로 넣으면 되고,
여기 있는 어떤 파일도 손으로 직접 고치지 않습니다.

**파이프라인이 라벨을 붙여 주기를 원하는, 아직 라벨이 없는 데이터** 는 `data/raw/` 안에, 녹화
세션별로 날짜 이름의 폴더를 만들어 넣습니다:

```
data/raw/20260114/some_recording.csv
data/raw/20260114/another_recording.csv
```

**분류기를 학습시킬, 라벨이 붙은 데이터** 는 `data/labeled/` 안에 **rev** 단위로 넣습니다. rev
하나는 "한 피험자를 하루 동안 녹화한 것"을 뜻합니다. 파일 이름은
`annotated_loco_<rev>_trial_<번호>.csv` 형식을 따릅니다:

```
data/labeled/rev2/annotated_loco_rev2_trial_1.csv
data/labeled/rev3/annotated_loco_rev3_trial_1.csv
```

라벨이 있는 데이터와 없는 데이터는 반드시 분리하세요. 라벨이 붙은 파일을 `data/raw/`에 넣으면
안 됩니다.

(이런 이름으로 정리하는 코드 툴이 필요하시다면 seowonlpark@gmail.com으로 이메일 보내주세요)

**락박스(lockbox, 최종 시험용으로 잠가 두는 데이터).**
점수를 믿으려면, 일부 라벨 데이터를 **학습 전에** 따로 떼어 두고 맨 마지막까지 절대 들여다보지 
않아야 합니다. 그러기 위해서 나온 시스템은 이 락박스입니다! 모델이 학습한 사람들을 그저 외운 것이 
아니라, 처음 보는 새로운 사람에게도 진짜로 일반화되는지를 확인하는 방법입니다.

어떤 rev를 락박스로 쓸지는 [`stages/s2_ml/dataset.py`](stages/s2_ml/dataset.py) 파일의 한 줄로
정합니다:

```python
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")   # 최종 시험 때까지 잠가 둘 rev
```

모델이 절대 학습하지 않을 rev(피험자)를 한둘 여기에 넣으세요. 파이프라인이 코드 수준에서 이를
강제합니다.

> **주의:** 락박스는 **한 번만 쓸 수 있습니다.** 최종 숫자를 얻기 위해 한 번 열면 그것으로 소진되며,
> 정직하게 다시 시험하려면 새로운 rev를 따로 떼어 두어야 합니다.

---

## 실행 (명령어 하나)

```powershell
python run_pipeline.py
```

이 한 줄이 결정론적 파이프라인 전체를 순서대로 돌립니다 - 데이터 정제, 분류기 학습과 채점, 물리
계산, 그리고 최종 결합 리포트까지 - 전부 **무료**로 실행하며, 무언가 빠져 있으면 명확한 안내
메시지와 함께 멈춥니다.

네 개의 AI 에이전트 단계까지 함께 돌리려면 (이때는 API 요금이 듭니다):

```powershell
python run_pipeline.py --with-agents
```

유용한 옵션: `--list`는 실행하지 않고 단계 목록만 보여 주고, `--dry-run`은 실제로 돌릴 명령들을
미리 보여 줍니다.

---

## 실행이 끝나면...

**최종 결과물.** 실행이 끝나면 대부분의 사람이 원하는 두 파일은 `runs/s4_fusion/` 아래에 생깁니다:

- **`fusion_report.md`** - 사람이 읽을 수 있는 요약본입니다. 결합된 판정이 얼마나 잘 맞았는지,
  그리고 신뢰도 등급별로 데이터를 얼마나 라벨링할 수 있는지를 보여 줍니다.
- **`fused_windows.csv`** - 구간마다 한 줄씩, 판정(서 있기 / 걷기)과 그 신뢰도 등급(**high** /
  **medium** / **low**)이 들어 있습니다.

이 프로젝트의 핵심은 바로 그 마지막 신뢰도 값입니다. **high** 나 **medium** 판정은 믿고 따라도
되는 판정이고, **low** 판정은 시스템이 정직하게 "여기는 확신이 없다"고 말하는 것입니다 - 추측이
아니라 판단 보류입니다. 실제 기기라면, 확신하는 판정에만 반응하고 신뢰도가 낮은 판정에서는
움직임을 멈추는 식으로 쓰게 됩니다.

판정을 원본 파일에 다시 써 넣고 싶다면 (녹화 파일마다 라벨과 신뢰도가 열로 추가된 CSV 한 개씩),
`python -m stages.s4_fusion.export` 를 실행하세요. 결과는 `results/` 폴더에 생깁니다. 파이프라인이
채점하지 않은 행(전환 구간, 버려지거나 너무 짧은 구간, 봉인된 lockbox 에 속한 행)은 일부러 빈칸으로
둡니다. export 는 계산하지 않은 값을 절대 추측해서 채우지 않습니다.

### 결과물의 모든 형태

이 파이프라인은 보고서 하나가 아니라 여러 개가 쌓인 것입니다 - 각 단계가 자기 결과물을 쓰며, 그
결과물은 몇 가지 정해진 형태로 나옵니다. 이 파일들은 손으로 고치지 않으며, 다시 실행하면 전부
새로 만들어집니다. 아래는 전체 목록으로, 각 형태와 그 형태가 무엇을 위한 것인지에 따라 묶었습니다.

무료 실행(`python run_pipeline.py`)은 AI 에이전트 결과물을 뺀 나머지 전부를 만듭니다. 아래에
**(에이전트)** 로 표시된 항목은 에이전트 단계까지 함께 실행할 때(`--with-agents`)만 생기며,
lockbox 결과는 의도적으로 단 한 번만 하는 실행에서만 생깁니다. 그렇게 하지 않으면 해당 파일은
그냥 없을 뿐입니다 - 무엇도 지어내지 않습니다.

**사람이 읽는 보고서 (`.md`)** - 바로 읽으라고 만든 서술형 요약본입니다:

- `runs/s1_census/census.md` - 코퍼스 점검: 파일과 계열 수, 그리고 예상치 못한 열 이름.
- `runs/s1_clean/clean_report.md` - 정제 집계: 사용 가능 대 격리된 파일, 남은 분량(분), 레이트 구성,
  채널별 신뢰 플래그.
- `runs/s2_ml/locoeval.md` - 학습된 분류기가 rev 별로 얼마나 잘 맞추는지.
- `runs/s4_fusion/fusion_report.md` - 위에서 설명한 핵심 결과물.
- `runs/s4_fusion/curation.md` **(에이전트)** - 재라벨 / 새 클래스 / 추가 수집으로 보낼 플래그된
  구간의 대기열.
- `runs/s4_fusion/new_class_candidates.md` **(에이전트)** - 두 라벨 분류 체계가 놓쳤을 수 있는 구조로,
  각 항목이 사람 검토용으로 표시됩니다.
- `runs/s4_fusion/lockbox_result.md` **(lockbox 전용)** - 단 한 번 쓰는 봉인 세트 점수로, lockbox 를
  의도적으로 열 때만 기록됩니다.

**데이터 표 (`.csv` / `.parquet`)** - 레코드마다 한 줄씩, 코드나 스프레드시트로 불러오라고 만든 것입니다:

- `runs/s4_fusion/fused_windows.csv` - 채점된 구간마다 한 줄 (핵심 결과 표).
- `results/<recording>.csv` - 원본 녹화 파일에 결합 라벨과 신뢰도가 덧붙은 것 (위에서 설명한 export).
- `runs/s3_physics/anchors.csv` - 구간별 물리 앵커 측정값.
- `runs/s2_ml/oof_champion.csv` - 분류기의 out-of-fold 예측값 (fusion 이 물리 관점과 결합하는 입력).
- `data/clean/**.parquet` - 정제되고 리샘플링된 센서 데이터 그 자체, 녹화 파일마다 하나씩.

**기계가 읽는 상태와 지표 (`.json`)** - 코드가 소비할 정확한 수치와 설정값입니다:

- `runs/s2_ml/champion.json`, `model_meta.json`, `taxonomy.json`, `locoeval.json` - 우승 모델의 정의,
  학습 메타데이터, 클래스 분류 체계, 전체 점수.
- `runs/s3_physics/rate_audit.json`, `disagreement.json` - 레이트 불변성 판정과 물리 대 라벨 불일치.
- `runs/s4_fusion/fusion.json`, `disagreements.json` - fusion 지표와 두 관점이 갈리는 사례.
- `data/clean/**.channel_trust.json` - 어떤 채널을 신뢰했는지 기록한 파일별 사이드카.

**추가만 하는 원장 (`.jsonl`)** - 한 줄에 JSON 레코드 하나씩, 덮어쓰지 않고 쌓여 가는 감사 기록입니다:

- `runs/s1_*/manifest.jsonl`, `segments.jsonl`, `observations.jsonl`, `quarantine.jsonl` - 무엇이
  들어왔고, 유지됐고, 관찰됐고, 거부됐는지.
- `runs/s2_ml/experiments.jsonl`, `proposals.jsonl` - 시도한 모든 모델 아이디어와 거부된 모든 아이디어
  (거부 기록이 같은 아이디어의 재제안을 막아 줍니다).
- `runs/s3_physics/hypotheses.jsonl`, `runs/s4_fusion/fusion_review.jsonl`,
  `runs/s4_fusion/curation_queue.jsonl` **(에이전트)** - 에이전트가 남긴 추론과 분류 결정.

**그래프 (`.png`)** - 물리 에이전트가 읽는 시각 자료입니다:

- `runs/s3_physics/plots/trial_*.png` - 시험마다 그림 하나씩. `rate_audit.png` - 레이트 불변성 개요.

**학습된 모델 (`.joblib`)** - `runs/s2_ml/champion.joblib`, 저장된 분류기 그 자체로, 새 데이터를 바로
채점할 수 있습니다.

**실행별 에이전트 로그 (에이전트)** - 각 에이전트 단계는 프롬프트, 도구 로그, 비용을 담은 타임스탬프
폴더 `runs/<date>_runN/` 도 함께 씁니다. 그래서 어떤 AI 판단이든 그것이 실제로 무엇을 봤는지까지
되짚을 수 있습니다.

단계별 명령어-결과물 대응 전체 목록은 [`PIPELINE.md`](PIPELINE.md)의 8-9번 항목(영어)을 참고하세요.

---

## 단계별로 실행하기

보통은 위의 명령어 하나면 충분합니다. 더 세밀하게 다루고 싶다면, 각 단계를 따로 실행할 수도
있습니다:

| 명령어 | 하는 일 |
|---|---|
| `python -m stages.s1_clean.clean --out runs/s1_clean` | 원본 데이터 정제 + 리샘플링 |
| `python -m stages.s2_ml.train --out runs/s2_ml --taxonomy` | 분류기 학습 + 채점 |
| `python -m stages.s3_physics.run` | 물리 앵커 + 그래프 계산 |
| `python -m stages.s4_fusion.run` | 결합 판정 + 신뢰도 산출 |
| `python orchestrator.py s1_exception` | AI: 이상한 파일 분류 (요금 발생) |
| `python orchestrator.py s3_physics` | AI: 그래프를 읽고 규칙 제안 (요금 발생) |
| `python orchestrator.py s4_fusion` | AI: 의견이 갈리는 사례 검토 (요금 발생) |

전체 목록은 [`PIPELINE.md`](PIPELINE.md)의 8번 항목(영어)을 참고하세요.

---

## 더 알아보기

- [`PIPELINE.md`](PIPELINE.md) - 각 부분이 어떻게 동작하는지, 그리고 정확한 명령어와 결과물. (영어)
- [`BUILDLOG.md`](BUILDLOG.md) - 이 파이프라인을 왜 이렇게 만들었는지, 결정 하나하나. (영어)

문의: **seowonlpark@gmail.com**

디버깅하는 과정에서 도움이 될수있는 repo:
**https://github.com/seowonlpark/hurotics-imu-csv-validation**

---

## 문제 해결

- **"ModuleNotFoundError" / 패키지를 못 찾음** - 가상 환경이 켜져 있지 않습니다.
  `.venv\Scripts\Activate.ps1` 을 실행하세요 (새 터미널마다 매번 해야 합니다). 그다음 다시 실행합니다.
- **에이전트 단계에서 API 키 관련 오류가 남** - `.env` 파일이 없거나 키가 비어 있습니다.
  `.env.example` 을 `.env` 로 복사하고 키를 넣으세요. (에이전트가 아닌 단계는 키가 필요 없습니다.)
- **시작하자마자 "no raw data" 라고 나옴** - `data/raw/` 가 비어 있습니다. 먼저
  `data/raw/<날짜>/` 아래에 녹화 파일을 넣으세요.
- **PowerShell이 활성화 스크립트를 실행하지 못함** -
  `Set-ExecutionPolicy -Scope Process RemoteSigned` 을 한 번 실행한 뒤 다시 켜세요.
