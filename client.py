"""Code Review Template Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CodeReviewTemplateAction, CodeReviewTemplateObservation
except ImportError:
    from models import CodeReviewTemplateAction, CodeReviewTemplateObservation


class CodeReviewTemplateEnv(
    EnvClient[CodeReviewTemplateAction, CodeReviewTemplateObservation, State]
):
    def _step_payload(self, action: CodeReviewTemplateAction) -> Dict:
        return {
            "issues": action.issues,
            "quality_score": action.quality_score,
            "suggestion": action.suggestion,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CodeReviewTemplateObservation]:
        obs_data = payload.get("observation", {})

        # 🔒 Clamp score strictly within (0,1)
        raw_score = obs_data.get("score", 0.01)
        score = max(min(raw_score, 0.95), 0.05)

        # 🔒 Clamp reward strictly within (0,1)
        raw_reward = payload.get("reward", 0.01)
        reward = max(min(raw_reward, 0.95), 0.05)

        observation = CodeReviewTemplateObservation(
            code=obs_data.get("code", ""),
            task_type=obs_data.get("task_type", "easy"),
            feedback=obs_data.get("feedback", ""),
            score=score,
            done=payload.get("done", False),
            reward=reward,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )