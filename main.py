from ai_scientist import AIScientist, DEFAULT_PROMPTS
from prompt_optimizer_simple import SimplePromptOptimizer, save_optimized_prompts
import os


def run_basic_scientist(research_topic):
    """Run AI scientist with default prompts"""
    print("\n" + "="*80)
    print("Running AI Scientist with Default Prompts")
    print("="*80 + "\n")

    scientist = AIScientist(model_name="gpt-4o-mini")

    result = scientist.run(
        DEFAULT_PROMPTS["researcher"],
        DEFAULT_PROMPTS["analyst"],
        DEFAULT_PROMPTS["writer"],
        research_topic
    )

    print("\n" + "="*80)
    print("AI Scientist Result:")
    print("="*80)
    print(result)
    print("="*80 + "\n")

    return result


def run_optimized_scientist(research_topic, iterations=3):
    """Run AI scientist with optimized prompts"""
    print("\n" + "="*80)
    print("Optimizing AI Scientist Prompts")
    print("="*80 + "\n")

    # Initialize optimizer
    optimizer = SimplePromptOptimizer(model_name="gpt-4o")

    # Optimize prompts
    optimized_prompts, score = optimizer.optimize(
        research_topic=research_topic,
        iterations=iterations
    )

    # Save optimized prompts
    save_optimized_prompts(optimized_prompts)

    print("\n" + "="*80)
    print("Optimized Prompts:")
    print("="*80)
    for agent_name, prompts in optimized_prompts.items():
        print(f"\n{agent_name.upper()}:")
        print(f"  Goal: {prompts['goal']}")
        print(f"  Backstory: {prompts['backstory']}")
    print("="*80 + "\n")

    return optimized_prompts, score


def main():
    # Set your research topic
    RESEARCH_TOPIC = "The impact of artificial intelligence on scientific research productivity"

    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your-api-key-here")
        return

    print("\n" + "="*80)
    print("AI Scientist with Prompt Optimization")
    print("="*80)

    # Option 1: Run basic AI scientist with default prompts
    print("\nOption 1: Run with default prompts")
    print("Option 2: Run with prompt optimization")
    print("Option 3: Run both")

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == "1":
        # Run with default prompts
        run_basic_scientist(RESEARCH_TOPIC)

    elif choice == "2":
        # Run with optimization
        iterations = int(input("Number of optimization iterations (default 5): ").strip() or "5")
        run_optimized_scientist(RESEARCH_TOPIC, iterations=iterations)

    elif choice == "3":
        # Run both
        print("\n--- Running with default prompts first ---")
        run_basic_scientist(RESEARCH_TOPIC)

        print("\n--- Now optimizing prompts ---")
        iterations = int(input("Number of optimization iterations (default 5): ").strip() or "5")
        run_optimized_scientist(RESEARCH_TOPIC, iterations=iterations)

    else:
        print("Invalid option. Exiting.")


if __name__ == "__main__":
    main()