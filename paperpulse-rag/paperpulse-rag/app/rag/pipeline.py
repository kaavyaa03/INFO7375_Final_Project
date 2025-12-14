def build_context(docs):
    context_lines = []
    citations = []
    for d in docs:
        cid = d.metadata.get("chunk_id", "na")
        src = d.metadata.get("source", "unknown")
        context_lines.append(f"[{cid}] {d.page_content}")
        citations.append({"source": src, "chunk_id": cid})
    return "\n".join(context_lines), citations

def answer_question(question: str, mode: str, docs):
    if not docs:
        return "I don’t have enough evidence in the knowledge base to answer this question.", []

    citations = []
    extracted_points = []

    for d in docs:
        cid = d.metadata.get("chunk_id", "na")
        src = d.metadata.get("source", "unknown")
        citations.append({"source": src, "chunk_id": cid})

        # Take a short, safe snippet from each chunk
        snippet = d.page_content.strip().split(".")[0]
        extracted_points.append(f"- {snippet.strip()}.")

    citation_text = " ".join(
        [f"[source: {c['source']}, chunk: {c['chunk_id']}]" for c in citations]
    )

    if mode == "summarize":
        answer = (
            "Based on the retrieved documents, the following points are supported:\n\n"
            + "\n".join(extracted_points)
            + "\n\n"
            + citation_text
        )
    else:
        answer = (
            "The retrieved knowledge base indicates the following:\n\n"
            + "\n".join(extracted_points)
            + "\n\n"
            + citation_text
        )

    return answer, citations

