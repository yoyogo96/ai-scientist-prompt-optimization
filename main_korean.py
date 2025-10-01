#!/usr/bin/env python3
"""
한글 AI 과학자 프롬프트 최적화 메인 스크립트
"""
import os
from dotenv import load_dotenv
from prompt_optimizer_korean import KoreanPromptOptimizer, save_optimized_prompts_kr

# .env 파일에서 환경변수 로드
load_dotenv()


def run_korean_optimization(research_topic, iterations=10):
    """한글로 AI 과학자 최적화 실행"""
    optimizer = KoreanPromptOptimizer(model_name="gpt-4o")

    print(f"\n{'='*80}")
    print(f"🇰🇷 한글 AI 과학자 프롬프트 최적화")
    print(f"{'='*80}")
    print(f"연구 주제: {research_topic}")
    print(f"반복 횟수: {iterations}")
    print(f"평가 모델: GPT-4o (엄밀한 평가)")
    print(f"에이전트 모델: GPT-4o-mini (비용 효율적)")
    print(f"평가 척도: 0-100점 (소수점 포함)")
    print(f"{'='*80}\n")

    optimized_prompts, best_score = optimizer.optimize(
        research_topic=research_topic,
        iterations=iterations,
        output_dir="optimization_results_korean"
    )

    # 최적화된 프롬프트 저장
    save_optimized_prompts_kr(optimized_prompts, "optimized_prompts_korean.py")

    print(f"\n{'='*80}")
    print(f"✨ 한글 최적화 완료!")
    print(f"{'='*80}")
    print(f"최고 점수: {best_score:.1f}/100")
    print(f"결과 위치: optimization_results_korean/")
    print(f"최적화 프롬프트: optimized_prompts_korean.py")
    print(f"{'='*80}\n")

    return optimized_prompts, best_score


if __name__ == "__main__":
    # 연구 주제 (한글)
    research_topic = "인공지능이 과학 연구 생산성에 미치는 영향"

    # 10회 반복 최적화 실행
    run_korean_optimization(research_topic, iterations=10)
