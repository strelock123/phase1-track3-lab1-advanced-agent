ACTOR_SYSTEM = """
You are a precise question-answering agent for HotpotQA-style multi-hop questions.
Use only the provided context and the reflection memory, if any.
Reason silently and answer with a short final answer only.
Do not explain your reasoning unless explicitly asked.
"""

EVALUATOR_SYSTEM = """
You are a strict binary evaluator for a question answering task.
Compare the candidate answer against the gold answer and the provided context.
Return only valid JSON with the keys:
- score: 1 if the answer is correct after normalization, otherwise 0
- reason: a short explanation
- missing_evidence: a list of missing facts or hops
- spurious_claims: a list of unsupported or wrong claims
- final_answer: the candidate answer you evaluated
"""

REFLECTOR_SYSTEM = """
You are a reflection module for an iterative question answering agent.
Analyze the failure and produce a compact lesson that can help the next attempt.
Return only valid JSON with the keys:
- attempt_id: the failed attempt number
- failure_reason: the evaluator reason
- lesson: what the agent should learn from the mistake
- next_strategy: a concrete strategy for the next attempt
"""

def format_context(context: list[dict[str, str]] | list[object]) -> str:
    lines: list[str] = []
    for item in context:
        title = getattr(item, "title", None) or item["title"]
        text = getattr(item, "text", None) or item["text"]
        lines.append(f"- {title}: {text}")
    return "\n".join(lines)

def build_actor_user_prompt(question: str, context: list[object], reflection_memory: list[str]) -> str:
    memory_block = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "(none)"
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{format_context(context)}\n\n"
        f"Reflection memory:\n{memory_block}\n\n"
        "Answer with the final short answer only."
    )

def build_evaluator_user_prompt(question: str, gold_answer: str, candidate_answer: str, context: list[object]) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Context:\n{format_context(context)}\n\n"
        "Return JSON only."
    )

def build_reflector_user_prompt(question: str, gold_answer: str, answer: str, judge_reason: str, context: list[object], attempt_id: int) -> str:
    return (
        f"Attempt id: {attempt_id}\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Evaluator reason:\n{judge_reason}\n\n"
        f"Context:\n{format_context(context)}\n\n"
        "Return JSON only."
    )
