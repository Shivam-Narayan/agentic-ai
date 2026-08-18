from src.kt_agent.workflow import KnowledgeTransferAgent


def main() -> None:
    agent = KnowledgeTransferAgent()
    question = "What is the tech stack used in the project, explain in detail"
    final_output = None

    for output in agent.run(question):
        final_output = output

    print("\nFinal answer:")
    print(final_output.get("generation", "") if final_output else "")


if __name__ == "__main__":
    main()
