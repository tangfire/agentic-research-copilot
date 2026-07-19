from __future__ import annotations

import hashlib

from ..schemas import PlanItem, ResearchRequest


class PlannerAgent:
    def create_plan(self, request: ResearchRequest) -> list[PlanItem]:
        base = request.topic.strip()
        seed = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]

        questions = [
            ("problem", f"What is the core problem behind {base}?"),
            ("approach", f"Which agent patterns solve {base}?"),
            ("data", f"What data or knowledge sources are needed for {base}?"),
            ("risk", f"What are the failure modes of {base}?"),
        ]

        if request.depth == "deep":
            questions.append(("eval", f"How should {base} be evaluated?"))

        return [
            PlanItem(
                id=f"{seed}-{name}",
                question=question,
                purpose=f"Support a {request.depth} research brief",
            )
            for name, question in questions
        ]

