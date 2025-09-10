<h1 align="center">🚀 AICON</h1>
<h3 align="center">(아이:콘) — <i>AI C(see) Over the New</i></h3>

<p align="center">
  <b>Persona Simulation & Evidence-based Evaluation Framework</b><br/>
  AI Agent · Mi:dm 2.0 · Persona-driven Evaluation
</p>

---
## 📖 Overview

**AICON (아이:콘)**은 *Persona Simulation*과 *Evidence-based Evaluation*을 결합한 PDCA(Plan-Data-Check-Act) Cycle 자동 프레임워크입니다.  
외부 검색 결과(`evidence.json`)와 라벨링된 페르소나 대화 데이터(`data/shopping/*.json`)를 활용하여,  
후보 상품·요금제(`candidates.json`)에 대해 **시뮬레이션 기반 평가**를 수행할 수 있습니다.

이 프로젝트는 다음과 같은 목적을 가집니다:
- 사용자 페르소나별 응답 시뮬레이션
- 검색 결과를 통합한 **근거 기반 의사결정**
- 상품·요금제 후보에 대한 **자동 평가 및 비교**
---
## 📂 Project Structure

```text
data/shopping/
└── *.json           : 라벨링된 페르소나 대화 데이터

inputs/
├── evidence.json    : 외부 근거 데이터 (Summary of Search Agent Results)
└── candidates.json  : 평가할 후보 상품/요금제 목록

run_agent.py         : 실행 파일

```
---

## ⚡ Getting Started

### 1. vLLM으로 Midm-2.0-Base 서빙하기
```bash
python -m vllm.entrypoints.openai.api_server \
  --model K-intelligence/Midm-2.0-Base-Instruct \
  --port 8000 \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --dtype float16

curl http://localhost:8000/v1/models
```

### 2. 환경 변수 세팅
```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
```

### 3. Agent 실행
```bash
python run_agent.py
```
---
## 🛠️ Tech Stack

- vLLM · Python 3.10+

- CUDA-enabled GPU(권장: 40GB VRAM 이상)

- **K-intelligence/Midm-2.0-Base-Instruct**
---
## 📚 Data & References

- **Dataset**
  - **페르소나 대화**, 한국지능정보사회진흥원(NIA)
  - https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8E%98%EB%A5%B4%EC%86%8C%EB%82%98&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71302

- **Reference Paper**
  - **Using LLMs for Market Research**, James Brand, Ayelet Israeli, and Donald Ngwe  
  - [Harvard Business School Faculty & Research](https://www.hbs.edu/faculty/Pages/item.aspx?num=63859)
---
## License

- MIT License.

