import os
import json
from datetime import datetime
from ai_scientist import AIScientist, DEFAULT_PROMPTS
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SimplePromptOptimizer:
    def __init__(self, model_name="gpt-4o"):
        """Initialize Simple Prompt Optimizer without TextGrad"""
        self.model_name = model_name

    def evaluate_output(self, result):
        """Evaluate the quality of AI scientist output with fine-grained 0-100 scoring"""
        evaluation_prompt = f"""You are an expert scientific reviewer. Evaluate the following AI scientist's output with PRECISE, GRANULAR scoring.

Use a 0-100 scale where:
- 0-20: Severely deficient, unusable
- 21-40: Poor quality, major issues
- 41-60: Mediocre, significant improvements needed
- 61-75: Acceptable, but clear room for improvement
- 76-85: Good quality, minor improvements possible
- 86-95: Excellent, high-quality work
- 96-100: Outstanding, publication-ready

Evaluate on these dimensions (weight each equally):

1. **Relevance (0-100)**: How directly and comprehensively does it address the topic?
2. **Depth of Analysis (0-100)**: How thorough, detailed, and insightful is the analysis? Are specific examples, case studies, and quantitative data provided?
3. **Clarity (0-100)**: How clear, well-structured, and readable is the writing?
4. **Scientific Rigor (0-100)**: How sound is the methodology? Are claims supported by citations and evidence?
5. **Comprehensiveness (0-100)**: How complete is the coverage? Are multiple perspectives, ethical considerations, and limitations addressed?

Provide scores for EACH dimension, then calculate the OVERALL score as the average.

**CRITICAL**: Be precise with decimals. Use scores like 67.5, 72.3, 84.8, not just whole numbers.
**CRITICAL**: Identify specific weaknesses and strengths to justify the score.
**CRITICAL**: Be discerning - reserve scores above 85 for truly exceptional work.

Format your response EXACTLY as:
```
Relevance: [score]/100
Depth: [score]/100
Clarity: [score]/100
Rigor: [score]/100
Comprehensiveness: [score]/100
Overall: [score]/100

Feedback:
[Detailed feedback with specific examples of strengths and weaknesses for each dimension]
```

Output to evaluate:
{result}
"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a rigorous scientific reviewer who provides precise, fine-grained evaluations. Use decimal precision in your scores."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent evaluation
        )

        evaluation = response.choices[0].message.content

        # Extract overall score
        try:
            lines = evaluation.split('\n')
            for line in lines:
                if line.startswith('Overall:'):
                    score_text = line.split(':')[1].strip()
                    # Extract number before /100
                    score = float(score_text.split('/')[0].strip())
                    return score, evaluation
        except Exception as e:
            print(f"Warning: Could not parse score: {e}")
            pass

        return 50.0, evaluation

    def improve_prompt(self, current_prompt, feedback, role):
        """Use GPT-4 to improve prompts based on feedback"""
        improvement_prompt = f"""You are an expert prompt engineer specializing in AI agent optimization.

Role: {role}
Current Prompt:
Goal: {current_prompt['goal']}
Backstory: {current_prompt['backstory']}

Recent Evaluation Feedback:
{feedback}

Your task: Create SIGNIFICANTLY IMPROVED prompts that address the weaknesses mentioned in the feedback.

Key improvements needed:
- Add specific examples and case studies
- Include detailed methodological guidance
- Emphasize scientific rigor and citation practices
- Expand depth of analysis expectations
- Include ethical considerations
- Make the agent more proactive and comprehensive

Create detailed, professional prompts that will guide this agent to produce higher quality scientific work.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "goal": "detailed, specific goal that includes what to prioritize and how to approach the task",
  "backstory": "comprehensive backstory that establishes expertise, methods, and high standards for this role"
}}
"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in prompt engineering for AI research agents. Output only valid JSON."},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.8
        )

        try:
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            improved = json.loads(content)
            print(f"   ✓ Improved {role} prompts")
            return improved
        except Exception as e:
            print(f"   ⚠ Failed to parse improved prompt for {role}: {e}")
            print(f"   Using current prompt instead")
            return current_prompt

    def save_iteration_results(self, iteration, prompts, result, score, feedback, output_dir="optimization_results"):
        """Save iteration results to files"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save prompts
        prompts_file = f"{output_dir}/iteration_{iteration}_prompts.json"
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        # Save result
        result_file = f"{output_dir}/iteration_{iteration}_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Score: {score:.1f}/100\n")
            f.write(f"\n{feedback}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"AI Scientist Output:\n")
            f.write(f"{'='*80}\n\n")
            f.write(str(result))

        print(f"✓ Saved iteration {iteration} results to {output_dir}/")

    def optimize(self, research_topic, iterations=5, output_dir="optimization_results"):
        """Optimize prompts iteratively to maximize performance"""

        print(f"\n{'='*80}")
        print(f"Starting Aggressive Prompt Optimization for: {research_topic}")
        print(f"Goal: Maximize performance across {iterations} iterations")
        print(f"{'='*80}\n")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Start with default prompts
        current_prompts = {
            "researcher": DEFAULT_PROMPTS["researcher"].copy(),
            "analyst": DEFAULT_PROMPTS["analyst"].copy(),
            "writer": DEFAULT_PROMPTS["writer"].copy()
        }

        best_score = 0
        best_prompts = None
        all_iterations = []
        score_improvements = []

        scientist = AIScientist(model_name="gpt-4o-mini")

        for iteration in range(iterations):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*80}\n")

            # Run AI scientist with current prompts
            print("🔬 Running AI Scientist with current prompts...\n")
            result = scientist.run(
                current_prompts["researcher"],
                current_prompts["analyst"],
                current_prompts["writer"],
                research_topic
            )

            # Evaluate output
            print("\n📊 Evaluating output quality...")
            score, feedback = self.evaluate_output(result)

            print(f"\n✅ Current Score: {score:.1f}/100")

            # Calculate improvement
            if iteration > 0:
                improvement = score - all_iterations[-1]["score"]
                score_improvements.append(improvement)
                print(f"📈 Improvement: {improvement:+.1f} from previous iteration")

            print(f"\n📝 Feedback:\n{feedback}\n")

            # Save iteration results
            self.save_iteration_results(
                iteration + 1,
                current_prompts,
                result,
                score,
                feedback,
                output_dir
            )

            # Track all iterations
            all_iterations.append({
                "iteration": iteration + 1,
                "score": score,
                "prompts": current_prompts.copy()
            })

            # Track best prompts
            if score > best_score:
                best_score = score
                best_prompts = current_prompts.copy()
                print(f"🎯 New best score: {best_score:.1f}/100 (Best so far!)")
            else:
                print(f"📊 Best score remains: {best_score:.1f}/100")

            # Always improve prompts for next iteration (except last)
            if iteration < iterations - 1:
                print(f"\n🔧 Aggressively improving prompts for next iteration...")
                print(f"   Analyzing weaknesses and optimizing all agent prompts...")

                # Improve each agent's prompt based on feedback
                current_prompts["researcher"] = self.improve_prompt(
                    current_prompts["researcher"],
                    feedback,
                    "Research Scientist"
                )
                current_prompts["analyst"] = self.improve_prompt(
                    current_prompts["analyst"],
                    feedback,
                    "Data Analyst"
                )
                current_prompts["writer"] = self.improve_prompt(
                    current_prompts["writer"],
                    feedback,
                    "Scientific Writer"
                )

                print("✓ All prompts improved for next iteration\n")

        # Calculate statistics
        avg_improvement = sum(score_improvements) / len(score_improvements) if score_improvements else 0
        total_improvement = all_iterations[-1]["score"] - all_iterations[0]["score"]

        # Save summary
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

        # Print final summary
        print(f"\n{'='*80}")
        print(f"🎉 Optimization Complete!")
        print(f"{'='*80}")
        print(f"Initial Score:     {all_iterations[0]['score']:.1f}/100")
        print(f"Final Score:       {all_iterations[-1]['score']:.1f}/100")
        print(f"Best Score:        {best_score:.1f}/100")
        print(f"Total Improvement: {total_improvement:+.1f} points")
        if score_improvements:
            print(f"Avg Improvement:   {avg_improvement:+.2f} per iteration")
        print(f"Total Iterations:  {len(all_iterations)}")
        print(f"\nResults saved to {output_dir}/")
        print(f"{'='*80}\n")

        return best_prompts, best_score


def save_optimized_prompts(prompts, filename="optimized_prompts.py"):
    """Save optimized prompts to a file"""
    with open(filename, "w") as f:
        f.write("# Optimized Prompts Generated by Prompt Optimization\n\n")
        f.write("OPTIMIZED_PROMPTS = ")
        f.write(repr(prompts))
        f.write("\n")
    print(f"✓ Optimized prompts saved to {filename}")