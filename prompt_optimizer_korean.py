import os
import json
from datetime import datetime
from ai_scientist import AIScientist
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 한글 기본 프롬프트 (의도적으로 매우 낮은 품질)
DEFAULT_PROMPTS_KR = {
    "researcher": {
        "goal": "연구해",
        "backstory": "너는 연구하는 사람이야."
    },
    "analyst": {
        "goal": "분석해",
        "backstory": "너는 분석하는 사람이야."
    },
    "writer": {
        "goal": "글 써",
        "backstory": "너는 글 쓰는 사람이야."
    }
}


class KoreanPromptOptimizer:
    def __init__(self, model_name="gpt-4o"):
        """한글 프롬프트 최적화기 초기화"""
        self.model_name = model_name

    def evaluate_output(self, result):
        """AI 과학자 출력물을 0-100점 척도로 세밀하게 평가"""
        evaluation_prompt = f"""당신은 전문 과학 논문 심사위원입니다. 다음 AI 과학자의 출력물을 정밀하고 세밀하게 평가하세요.

0-100점 척도를 사용하세요:
- 0-20: 심각한 결함, 사용 불가
- 21-40: 낮은 품질, 주요 문제 있음
- 41-60: 평범함, 상당한 개선 필요
- 61-75: 수용 가능하나 개선의 여지 많음
- 76-85: 좋은 품질, 약간의 개선 가능
- 86-95: 탁월함, 높은 품질
- 96-100: 뛰어남, 출판 준비 완료

다음 차원으로 평가하세요 (각각 동일한 가중치):

1. **관련성 (0-100)**: 주제를 얼마나 직접적이고 포괄적으로 다루는가?
2. **분석 깊이 (0-100)**: 얼마나 철저하고 상세하며 통찰력 있는가? 구체적인 사례, 케이스 스터디, 정량적 데이터가 제공되는가?
3. **명료성 (0-100)**: 얼마나 명확하고 잘 구조화되어 있으며 읽기 쉬운가?
4. **과학적 엄밀성 (0-100)**: 방법론이 얼마나 타당한가? 주장이 인용과 증거로 뒷받침되는가?
5. **포괄성 (0-100)**: 다양한 관점, 윤리적 고려사항, 한계점을 다루는가?

각 차원별 점수를 제공한 후, 전체 점수를 평균으로 계산하세요.

**중요**: 소수점을 정확하게 사용하세요. 67.5, 72.3, 84.8과 같은 점수를 사용하고, 정수만 사용하지 마세요.
**중요**: 점수를 정당화할 구체적인 강점과 약점을 파악하세요.
**중요**: 신중하게 평가하세요 - 85점 이상은 진정으로 뛰어난 작업에만 부여하세요.

다음 형식으로 정확히 응답하세요:
```
관련성: [점수]/100
깊이: [점수]/100
명료성: [점수]/100
엄밀성: [점수]/100
포괄성: [점수]/100
전체: [점수]/100

피드백:
[각 차원별 강점과 약점에 대한 구체적인 예시를 포함한 상세한 피드백]
```

평가할 출력물:
{result}
"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "당신은 정밀하고 세밀한 평가를 제공하는 엄격한 과학 심사위원입니다. 점수에 소수점을 사용하세요."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3
        )

        evaluation = response.choices[0].message.content

        # 전체 점수 추출
        try:
            lines = evaluation.split('\n')
            for line in lines:
                if line.startswith('전체:') or line.startswith('Overall:'):
                    score_text = line.split(':')[1].strip()
                    score = float(score_text.split('/')[0].strip())
                    return score, evaluation
        except Exception as e:
            print(f"경고: 점수 파싱 실패: {e}")
            pass

        return 50.0, evaluation

    def improve_prompt(self, current_prompt, feedback, role):
        """GPT-4를 사용해 피드백 기반으로 프롬프트 개선"""
        improvement_prompt = f"""당신은 AI 에이전트 최적화 전문 프롬프트 엔지니어입니다.

역할: {role}
현재 프롬프트:
목표: {current_prompt['goal']}
배경: {current_prompt['backstory']}

최근 평가 피드백:
{feedback}

당신의 작업: 피드백에서 언급된 약점을 해결하는 대폭 개선된 프롬프트를 생성하세요.

필요한 주요 개선사항:
- 구체적인 예시와 케이스 스터디 추가
- 상세한 방법론적 지침 포함
- 과학적 엄밀성과 인용 관행 강조
- 분석 깊이 기대치 확대
- 윤리적 고려사항 포함
- 에이전트를 더 적극적이고 포괄적으로 만들기

이 에이전트가 더 높은 품질의 과학적 작업을 생성하도록 안내할 상세하고 전문적인 프롬프트를 생성하세요.

다음 형식의 유효한 JSON으로만 응답하세요 (마크다운 없이, 추가 텍스트 없이):
{{
  "goal": "우선순위와 접근 방법을 포함한 상세하고 구체적인 목표",
  "backstory": "전문성, 방법론, 높은 기준을 확립하는 포괄적인 배경"
}}
"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "당신은 AI 연구 에이전트를 위한 프롬프트 엔지니어링 전문가입니다. 유효한 JSON만 출력하세요."},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.8
        )

        try:
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            improved = json.loads(content)
            print(f"   ✓ {role} 프롬프트 개선 완료")
            return improved
        except Exception as e:
            print(f"   ⚠ {role} 프롬프트 파싱 실패: {e}")
            print(f"   현재 프롬프트 유지")
            return current_prompt

    def save_iteration_results(self, iteration, prompts, result, score, feedback, output_dir="optimization_results_korean"):
        """반복 결과를 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 프롬프트 저장
        prompts_file = f"{output_dir}/iteration_{iteration}_prompts.json"
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        # 결과 저장
        result_file = f"{output_dir}/iteration_{iteration}_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"반복: {iteration}\n")
            f.write(f"시간: {timestamp}\n")
            f.write(f"점수: {score:.1f}/100\n")
            f.write(f"\n{feedback}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"AI 과학자 출력:\n")
            f.write(f"{'='*80}\n\n")
            f.write(str(result))

        print(f"✓ 반복 {iteration} 결과를 {output_dir}/에 저장했습니다")

    def optimize(self, research_topic, iterations=10, output_dir="optimization_results_korean"):
        """반복적으로 프롬프트를 최적화하여 성능 극대화"""

        print(f"\n{'='*80}")
        print(f"한글 프롬프트 최적화 시작: {research_topic}")
        print(f"목표: {iterations}회 반복을 통한 성능 극대화")
        print(f"{'='*80}\n")

        os.makedirs(output_dir, exist_ok=True)

        # 한글 기본 프롬프트로 시작
        current_prompts = {
            "researcher": DEFAULT_PROMPTS_KR["researcher"].copy(),
            "analyst": DEFAULT_PROMPTS_KR["analyst"].copy(),
            "writer": DEFAULT_PROMPTS_KR["writer"].copy()
        }

        best_score = 0
        best_prompts = None
        all_iterations = []
        score_improvements = []

        scientist = AIScientist(model_name="gpt-4o-mini")

        for iteration in range(iterations):
            print(f"\n{'='*80}")
            print(f"반복 {iteration + 1}/{iterations}")
            print(f"{'='*80}\n")

            print("🔬 현재 프롬프트로 AI 과학자 실행 중...\n")
            result = scientist.run(
                current_prompts["researcher"],
                current_prompts["analyst"],
                current_prompts["writer"],
                research_topic
            )

            print("\n📊 출력물 품질 평가 중...")
            score, feedback = self.evaluate_output(result)

            print(f"\n✅ 현재 점수: {score:.1f}/100")

            if iteration > 0:
                improvement = score - all_iterations[-1]["score"]
                score_improvements.append(improvement)
                print(f"📈 개선도: 이전 반복 대비 {improvement:+.1f}")

            print(f"\n📝 피드백:\n{feedback}\n")

            self.save_iteration_results(
                iteration + 1,
                current_prompts,
                result,
                score,
                feedback,
                output_dir
            )

            all_iterations.append({
                "iteration": iteration + 1,
                "score": score,
                "prompts": current_prompts.copy()
            })

            if score > best_score:
                best_score = score
                best_prompts = current_prompts.copy()
                print(f"🎯 새로운 최고 점수: {best_score:.1f}/100 (현재까지 최고!)")
            else:
                print(f"📊 최고 점수 유지: {best_score:.1f}/100")

            if iteration < iterations - 1:
                print(f"\n🔧 다음 반복을 위한 프롬프트 적극 개선 중...")
                print(f"   약점 분석 및 모든 에이전트 프롬프트 최적화 중...")

                current_prompts["researcher"] = self.improve_prompt(
                    current_prompts["researcher"],
                    feedback,
                    "연구 과학자"
                )
                current_prompts["analyst"] = self.improve_prompt(
                    current_prompts["analyst"],
                    feedback,
                    "데이터 분석가"
                )
                current_prompts["writer"] = self.improve_prompt(
                    current_prompts["writer"],
                    feedback,
                    "과학 작가"
                )

                print("✓ 다음 반복을 위한 모든 프롬프트 개선 완료\n")

        # 통계 계산
        avg_improvement = sum(score_improvements) / len(score_improvements) if score_improvements else 0
        total_improvement = all_iterations[-1]["score"] - all_iterations[0]["score"]

        # 요약 저장
        summary_file = f"{output_dir}/optimization_summary.json"
        summary = {
            "research_topic": research_topic,
            "total_iterations": len(all_iterations),
            "best_score": best_score,
            "initial_score": all_iterations[0]["score"],
            "final_score": all_iterations[-1]["score"],
            "total_improvement": total_improvement,
            "average_improvement_per_iteration": avg_improvement,
            "iterations": all_iterations
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 최종 요약 출력
        print(f"\n{'='*80}")
        print(f"🎉 최적화 완료!")
        print(f"{'='*80}")
        print(f"초기 점수:        {all_iterations[0]['score']:.1f}/100")
        print(f"최종 점수:        {all_iterations[-1]['score']:.1f}/100")
        print(f"최고 점수:        {best_score:.1f}/100")
        print(f"총 개선도:        {total_improvement:+.1f} 점")
        if score_improvements:
            print(f"평균 개선도:      {avg_improvement:+.2f} 점/반복")
        print(f"총 반복 횟수:     {len(all_iterations)}")
        print(f"\n결과가 {output_dir}/에 저장되었습니다")
        print(f"{'='*80}\n")

        return best_prompts, best_score


def save_optimized_prompts_kr(prompts, filename="optimized_prompts_korean.py"):
    """최적화된 프롬프트를 파일로 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# 프롬프트 최적화로 생성된 최적화된 한글 프롬프트\n\n")
        f.write("OPTIMIZED_PROMPTS_KR = ")
        f.write(repr(prompts))
        f.write("\n")
    print(f"✓ 최적화된 프롬프트가 {filename}에 저장되었습니다")
