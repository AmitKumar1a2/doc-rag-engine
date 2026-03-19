from rag_pipeline import ask_question


def main():
    print("Document RAG Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not question:
            continue

        try:
            result = ask_question(question)

            print("\nAssistant:")
            print(result["answer"])

            print("\nTop Sources:")
            for i, doc in enumerate(result["sources"], start=1):
                source = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", "unknown")
                print(f"{i}. {source} | page {page}")

            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()