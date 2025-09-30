# AI Scientist with TextGrad Optimization

CrewAI를 사용한 AI Scientist 시스템과 TextGrad를 이용한 프롬프트 최적화

## 구성

- **ai_scientist.py**: CrewAI를 사용한 AI Scientist (Researcher, Analyst, Writer 에이전트)
- **prompt_optimizer.py**: TextGrad를 사용한 프롬프트 최적화
- **main.py**: 메인 실행 스크립트

## 설치

```bash
pip install -r requirements.txt
```

## 설정

1. `.env` 파일 생성:
```bash
cp .env.example .env
```

2. OpenAI API 키 설정:
```
OPENAI_API_KEY=your-api-key-here
```

## 사용법

```bash
python main.py
```

실행 옵션:
- **Option 1**: 기본 프롬프트로 AI Scientist 실행
- **Option 2**: TextGrad로 프롬프트 최적화 후 실행
- **Option 3**: 둘 다 실행하여 비교

## 작동 방식

1. **AI Scientist (CrewAI)**:
   - Researcher: 주제에 대한 연구 수행
   - Analyst: 연구 결과 분석
   - Writer: 과학 보고서 작성

2. **TextGrad Optimization**:
   - AI Scientist의 출력 품질 평가
   - 그래디언트 계산을 통해 프롬프트 개선
   - 반복적으로 최적화하여 최상의 프롬프트 도출

## 출력

최적화된 프롬프트는 `optimized_prompts.py`에 저장됩니다.