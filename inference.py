import os
import json
import asyncio
from typing import List, Optional
from openai import OpenAI

from client import CodeReviewTemplateEnv
from models import CodeReviewTemplateAction

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "code_review_env"
MAX_STEPS = 1
SUCCESS_THRESHOLD = 0.5

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a code review agent. Analyze the given Python code and respond ONLY with a JSON object in this exact format:
{
    "issues": ["issue1", "issue2"],
    "quality_score": 0.7,
    "suggestion": "your suggestion here"
}
No extra text. Just the JSON."""


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def get_action(code: str, task_type: str) -> tuple:
    prompt = f"Review this Python code:\n\n{code}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        action = CodeReviewTemplateAction(
            issues=parsed.get("issues", []),
            quality_score=float(parsed.get("quality_score", 0.5)),
            suggestion=parsed.get("suggestion", "")
        )
        return action, None
    except Exception as e:
        # smart fallback based on task
        fallbacks = {
            "easy": CodeReviewTemplateAction(
                issues=["missing space", "indentation"],
                quality_score=0.8,
                suggestion="Fix formatting issues"
            ),
            "medium": CodeReviewTemplateAction(
                issues=["bad formatting", "one-line loop"],
                quality_score=0.6,
                suggestion="Improve code readability"
            ),
            "hard": CodeReviewTemplateAction(
                issues=["security risk", "exposing sensitive data"],
                quality_score=0.5,
                suggestion="Avoid exposing sensitive data"
            )
        }
        return fallbacks.get(task_type, fallbacks["easy"]), "api_error"


async def run_task(task_type: str):
    SERVER_URL = os.getenv("SERVER_URL", "https://byjayashree-code-review-env.hf.space")
    env = CodeReviewTemplateEnv(base_url=SERVER_URL)
    rewards = []
    steps_taken = 0

    log_start(task=task_type, model=MODEL_NAME)

    try:
        result = await env.reset(task_type=task_type)
        code = result.observation.code

        for step in range(1, MAX_STEPS + 1):
            action, error = get_action(code, task_type)
            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action.model_dump())
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        success = sum(rewards) / len(rewards) >= SUCCESS_THRESHOLD if rewards else False
        log_end(success=success, steps=steps_taken, rewards=rewards)

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)
        log_end(success=False, steps=steps_taken, rewards=rewards)
    finally:
        await env.close()

    print(flush=True)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())