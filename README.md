# AICON

## inputs/
  
- evidence.json
  - 외부 근거 데이터 (result of search_agent)

- candidates.json
  - list for eval(simulation)
  - 평가할 후보 상품/요금제 목록

## data/shopping/*.json

- 라벨링된 페르소나 대화 데이터

----------
## [사용방법]

### 1. vllm으로 Midm-2.0-Base 서빙

```
python -m vllm.entrypoints.openai.api_server \
  --model K-intelligence/Midm-2.0-Base-Instruct \
  --port 8000 \
  --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.8 --dtype float16
```
```
curl http://localhost:8000/v1/models
```
  - 잘 작동하는지 확인용 command

### 2. 환경변수 세팅

```
export OPENAI_API_BASE="http://localhost:8000/v1"
```
```
export OPENAI_API_KEY="dummy"
```


### 3. agent 파일 실행
```python persona_agent.py```



