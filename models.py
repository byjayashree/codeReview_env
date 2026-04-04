"""
Data models for the Code Review Environment.
"""

from typing import List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CodeReviewTemplateAction(Action):
    """Action for the Code Review environment."""
    issues: List[str] = Field(default=[], description="List of detected issues in the code")
    quality_score: float = Field(default=0.0, description="Code quality score between 0 and 1")
    suggestion: str = Field(default="", description="Improvement suggestion for the code")


class CodeReviewTemplateObservation(Observation):
    """Observation from the Code Review environment."""
    code: str = Field(default="", description="Python code snippet to review")
    task_type: str = Field(default="easy", description="Difficulty level: easy, medium, hard")
    feedback: str = Field(default="", description="Feedback on the agent's last action")
    score: float = Field(default=0.0, description="Score from the grader 0.0 to 1.0")