"""
Code Review Environment Implementation.
An agent reviews Python code and identifies issues, scores quality, and suggests fixes.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CodeReviewTemplateAction, CodeReviewTemplateObservation
except ImportError:
    from models import CodeReviewTemplateAction, CodeReviewTemplateObservation

TASKS = {
    "easy": [
        {
            "code": "def add(a,b):\n return a+b",
            "issues": ["missing space", "indentation"],
        },
        {
            "code": "x=5\ny=10\nprint(x+y)",
            "issues": ["missing spaces around operator"],
        }
    ],
    "medium": [
        {
            "code": "for i in range(10): print(i)",
            "issues": ["bad formatting", "one-line loop"],
        },
        {
            "code": "def foo(lst):\n    for i in range(len(lst)): print(lst[i])",
            "issues": ["use enumerate", "one-line loop"],
        }
    ],
    "hard": [
        {
            "code": "password = input()\nprint(password)",
            "issues": ["security risk", "exposing sensitive data"],
        },
        {
            "code": "import os\ncmd = input()\nos.system(cmd)",
            "issues": ["command injection", "security risk"],
        }
    ]
}


def grade(action: dict, task: dict) -> float:
    predicted_issues = action.get("issues", [])
    true_issues = task["issues"]

    match_count = sum(1 for issue in predicted_issues if issue in true_issues)
    issue_score = match_count / len(true_issues) if true_issues else 0.0

    predicted_quality = action.get("quality_score", 0)
    quality_score = min(max(float(predicted_quality), 0.0), 1.0)

    suggestion = action.get("suggestion", "").lower()
    keywords = ["secure", "avoid", "fix", "improve", "use", "replace", "remove"]
    suggestion_score = 1.0 if any(word in suggestion for word in keywords) else 0.0

    final_score = (
        0.5 * issue_score +
        0.2 * quality_score +
        0.3 * suggestion_score
    )

    return round(final_score, 2)


class CodeReviewTemplateEnvironment(Environment):
    """
    Code Review Environment where an agent evaluates Python code quality.
    Tasks range from easy formatting issues to hard security vulnerabilities.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = None
        self._task_type = "easy"

    def reset(self) -> CodeReviewTemplateObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # cycle through difficulty levels
        task_types = ["easy", "medium", "hard"]
        self._task_type = task_types[self._state.step_count % 3]
        self._current_task = random.choice(TASKS[self._task_type])

        return CodeReviewTemplateObservation(
            code=self._current_task["code"],
            task_type=self._task_type,
            feedback="",
            score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: CodeReviewTemplateAction) -> CodeReviewTemplateObservation:
        self._state.step_count += 1

        score = grade(action.model_dump(), self._current_task)

        feedback = f"Issues matched: {action.issues}. Score: {score}"

        return CodeReviewTemplateObservation(
            code=self._current_task["code"],
            task_type=self._task_type,
            feedback=feedback,
            score=score,
            done=True,
            reward=score,
            metadata={"step": self._state.step_count, "task_type": self._task_type},
        )

    @property
    def state(self) -> State:
        return self._state