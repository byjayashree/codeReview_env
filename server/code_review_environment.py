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
            "issues": ["bad formatting", "one-line loop", "one liner", "single line loop"],
        },
        {
            "code": "def foo(lst):\n    for i in range(len(lst)): print(lst[i])",
            "issues": ["use enumerate", "one-line loop", "one liner", "single line loop"],
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

# Strictly within (0, 1) — never 0.0 or 1.0
SCORE_MIN = 0.05
SCORE_MAX = 0.95


def clamp_strict(value: float) -> float:
    value = min(max(value, SCORE_MIN), SCORE_MAX)
    if value <= 0.0:
        value = SCORE_MIN
    if value >= 1.0:
        value = SCORE_MAX
    return round(value, 3)


def grade(action: dict, task: dict) -> float:
    predicted_issues = [i.lower() for i in action.get("issues", [])]
    true_issues = [i.lower() for i in task["issues"]]

    match_count = sum(
        1 for true in true_issues
        if any(true in pred or pred in true for pred in predicted_issues)
    )
    issue_score = match_count / len(true_issues) if true_issues else 0.1
    issue_score = max(issue_score, 0.1)

    # Clamp quality_score from agent to a safe range before using it
    raw_quality = float(action.get("quality_score", 0.5))
    quality_score = min(max(raw_quality, 0.05), 0.95)

    suggestion = action.get("suggestion", "").lower()
    keywords = [
        "secure", "avoid", "fix", "improve", "use", "replace", "remove",
        "space", "indent", "format", "pep", "injection", "sensitive", "password"
    ]
    suggestion_score = 0.9 if any(word in suggestion for word in keywords) else 0.1

    raw_score = 0.5 * issue_score + 0.2 * quality_score + 0.3 * suggestion_score

    # Guarantee score is strictly within (0, 1)
    return clamp_strict(raw_score)


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

    def reset(self, task_type: str = None, **kwargs) -> CodeReviewTemplateObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if task_type:
            self._task_type = task_type
        else:
            task_types = ["easy", "medium", "hard"]
            self._task_type = random.choice(task_types)
        self._current_task = random.choice(TASKS[self._task_type])

        return CodeReviewTemplateObservation(
            code=self._current_task["code"],
            task_type=self._task_type,
            feedback="",
            score=0.05,
            done=False,
            reward=0.05,
        )

def step(self, action: CodeReviewTemplateAction) -> CodeReviewTemplateObservation:
    self._state.step_count += 1
    score = grade(action.model_dump(), self._current_task)
    # Nuclear clamp - absolutely cannot be 0.0 or 1.0
    score = round(min(max(float(score), 0.05), 0.95), 4)
    if score <= 0.0: score = 0.05
    if score >= 1.0: score = 0.95

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