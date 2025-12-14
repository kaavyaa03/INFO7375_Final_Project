CITATION_RULES = """
Rules:
- Use ONLY retrieved context
- If context is insufficient, say: "I don’t have enough evidence in the knowledge base."
- Cite key claims with: [source: filename, chunk: id]
"""

QA_PROMPT = """
You are a research assistant.

{rules}

Context:
{context}

Question:
{question}

Answer with citations.
"""

SUMMARY_PROMPT = """
You are a summarization assistant.

{rules}

Context:
{context}

User request:
{question}

Return structured bullets with citations.
"""
