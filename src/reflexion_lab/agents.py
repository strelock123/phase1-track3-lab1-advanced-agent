from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, build_reflection, generate_actor_answer, judge_answer
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            actor_call = generate_actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, judge_call = judge_answer(example, actor_call.content)
            reflection_entry = None
            reflection_call_tokens = 0
            reflection_call_latency = 0
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_entry, reflection_call = build_reflection(example, attempt_id, actor_call.content, judge)
                reflection_memory.append(f"Attempt {attempt_id}: {reflection_entry.lesson} Next: {reflection_entry.next_strategy}")
                reflections.append(reflection_entry)
                reflection_call_tokens = reflection_call.usage.total_tokens
                reflection_call_latency = reflection_call.latency_ms
            token_estimate = actor_call.usage.total_tokens + judge_call.usage.total_tokens + reflection_call_tokens
            latency_ms = actor_call.latency_ms + judge_call.latency_ms + reflection_call_latency
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=actor_call.content,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = actor_call.content
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
