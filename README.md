# 💻 Code Review Environment — OpenEnv Hackathon Submission

## 🚀 Overview

This project implements a real-world reinforcement learning environment
for automated code review using the OpenEnv framework.

The environment simulates how an AI agent evaluates Python code —
detecting issues, scoring quality, and suggesting improvements.
It runs as a fully containerized FastAPI server, compliant with
the OpenEnv HTTP specification.

Code review is something every software team does daily. Training
agents to do it well has direct value in CI/CD pipelines, developer
tooling, and automated quality assurance.

---

## 🎯 Tasks

The environment defines 3 tasks with increasing difficulty:

### 🟢 Easy
Detect basic formatting issues — missing spaces, indentation problems.
The agent needs to identify surface-level style violations.

### 🟡 Medium
Identify structural and readability issues — one-line loops,
poor formatting patterns. The agent needs to reason about
code organization, not just style.

### 🔴 Hard
Detect security vulnerabilities — exposed sensitive data,
command injection risks. The agent needs to reason about
real-world consequences of bad code.

---

## 👁️ Observation Space

Each observation the agent receives contains:

- `code` — the Python code snippet to review
- `task_type` — difficulty level: easy, medium, or hard
- `feedback` — grader feedback from the previous step
- `score` — the score from the last grader evaluation

---

## 🎯 Action Space

The agent must return:

- `issues` — list of detected problems in the code
- `quality_score` — code quality rating between 0.0 and 1.0
- `suggestion` — improvement advice as a string

---

## 🧮 Reward Function

Rewards are non-binary and multi-dimensional. The grader scores
the agent across three components:

- Issue matching → 50% weight
  How many real issues the agent correctly identified

- Quality score → 20% weight
  The agent's self-assessed quality rating

- Suggestion relevance → 30% weight
  Whether the suggestion contains actionable keywords

This ensures the agent receives partial progress signals
throughout the episode — not just a binary win/lose at the end.

---

## ⚙️ OpenEnv Compliance

This environment fully implements the OpenEnv HTTP specification.

Endpoints:
- `POST /reset` — reset environment, get initial observation
- `POST /step` — submit action, get observation and reward
- `GET /state` — get current episode state
- `GET /health` — health check
- `GET /schema` — action and observation schemas
- `GET /metadata` — environment metadata

Validated with:
- `openenv validate` — local structure check ✅
- `openenv validate --url` — runtime check, 6/6 passed ✅

---

## 📊 Baseline Results

Baseline agent scores using deterministic fallback:

- Easy → 0.96 
- Medium → 0.42
- Hard → 0.40

The difficulty progression is intentional. Easy tasks are solvable
with basic pattern matching. Medium and hard tasks require genuine
reasoning about code structure and security vulnerabilities.

---

## 🐳 Setup and Usage

Run locally:

```bash
pip install openenv-core
uvicorn server.app:app --host 0.0.0.0 --port 8000
python inference.py
```

Environment variables the inference script reads:

```bash
HF_TOKEN=your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

Docker:

```bash
docker build -t code-review-env -f server/Dockerfile .
docker run -p 8000:8000 code-review-env
```

---

## 🗂️ Project Structure

```
codeReview_env/
├── server/
│   ├── app.py                        ← FastAPI server
│   ├── code_review_environment.py ← Environment logic and grader
│   ├── Dockerfile
│   └── requirements.txt
├── models.py      ← Pydantic Action and Observation models
├── client.py      ← EnvClient for inference script
├── inference.py   ← Baseline inference script
├── openenv.yaml   ← OpenEnv metadata
├── pyproject.toml ← Project config
└── README.md
```

---

## 🤖 Inference Script

The inference script connects to the running server as a client,
sends actions for each task, and emits structured logs:

```
[START] task=easy env=code_review_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.96 done=true error=null
[END] success=true steps=1 rewards=0.96
```

The script uses an OpenAI-compatible client with graceful fallback.
If the API is unavailable, a deterministic fallback agent ensures
reproducible baseline scores every time.

---

## ✨ Personal Note

This project was built under real pressure — practical exams
running back to back, a deadline that wasn't moving, and a
framework I had never touched before.

Honestly, that pressure made it interesting. There was no time
to overthink. I had to understand what OpenEnv actually was, how
a FastAPI server exposes RL environment endpoints, what Pydantic
models do, how a client connects to a server over WebSocket, and
why openenv validate was rejecting my code — all while debugging
import errors and indentation issues late at night.

This is the first time I built something that actually feels like
real AI infrastructure. Not a tutorial project. Not a copy-paste
script. An actual environment that runs as a server, accepts HTTP
requests, scores agent behavior, and passes a real validation suite.

I didn't know what a FastAPI app looked like before this.
I didn't know what a Pydantic model was.
I didn't know what uvicorn did.
Now I do — because I had to figure it out under pressure.

That's the part that matters more than the submission itself.

---
