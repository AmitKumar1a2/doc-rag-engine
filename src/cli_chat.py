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

            print("\nCitations:")
            sources = result.get("sources", {})
            if not sources:
                print("(No Citations found in the documents, Please refine your question.)")
            else:
                for citation_id, (source, page) in sources.items():
                    print(f"[{citation_id}] {source} | page {page}")

            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
