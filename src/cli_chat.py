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

            query_type = result.get("query_type", "UNKNOWN")
            routing = result.get("routing", {})
            route_source = routing.get("route_source", "UNKNOWN")
            confidence = routing.get("confidence", "n/a")
            reason = routing.get("reason", "")

            print(f"\nRoute: {query_type} ({route_source}, confidence={confidence})")
            if reason:
                print(f"Route reason: {reason}")

            print("\nAssistant:")
            print(result["answer"])

            print("\nCitations:")
            sources = result.get("sources", {})
            if not sources:
                if query_type == "CONVERSATION":
                    print("(not applicable: conversation route, no retrieval)")
                else:
                    print("(none)")
            else:
                for citation_id, (source, page) in sources.items():
                    print(f"[{citation_id}] {source} | page {page}")

            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
