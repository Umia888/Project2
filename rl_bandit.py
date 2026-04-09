"""
轻量级强化学习：文档检索排序的 ε-贪心多臂老虎机（Multi-Armed Bandit）。

说明：
- 非深度 RL（无神经网络策略、无 PPO/DQN）；属于在线 bandit / 增量式策略优化。
- 臂（arm）：llm_rerank = 调用 LLM 对文档编号排序；lexical_only = 仅用词重叠检索（省 API、可能略降相关性）。
- 奖励（reward）：主工作流内每次 Critic 判定的平均「通过率」（PASS=1，FAIL=0），含 replan 中的多次判定。
- 更新规则：样本平均（等价于步长 1/n 的增量均值），持久化到 JSON 便于跨会话学习。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

DEFAULT_ARMS = ("llm_rerank", "lexical_only")

_bandit_singleton: Optional["RerankBandit"] = None


class RerankBandit:
    def __init__(self, state_path: Path, epsilon: float = 0.12):
        self.state_path = state_path
        self.epsilon = epsilon
        self.counts: dict[str, int] = {}
        self.values: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.state_path.is_file():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.counts = {str(k): int(v) for k, v in (data.get("counts") or {}).items()}
                self.values = {str(k): float(v) for k, v in (data.get("values") or {}).items()}
                self.epsilon = float(data.get("epsilon", self.epsilon))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                self.counts = {}
                self.values = {}
        for a in DEFAULT_ARMS:
            self.counts.setdefault(a, 0)
            self.values.setdefault(a, 0.5)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "counts": self.counts,
            "values": self.values,
            "epsilon": self.epsilon,
            "arms": list(DEFAULT_ARMS),
            "note": "epsilon-greedy bandit for RAG reranking; reward = mean Critic PASS in episode",
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def select_arm(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(DEFAULT_ARMS))
        return max(DEFAULT_ARMS, key=lambda a: self.values.get(a, 0.0))

    def update(self, arm: str, reward: float) -> None:
        if arm not in DEFAULT_ARMS:
            return
        reward = max(0.0, min(1.0, float(reward)))
        self.counts[arm] = self.counts.get(arm, 0) + 1
        n = self.counts[arm]
        old = self.values.get(arm, 0.0)
        self.values[arm] = old + (reward - old) / n
        self._save()

    def snapshot(self) -> dict:
        return {
            a: {"n": self.counts.get(a, 0), "q": round(self.values.get(a, 0.0), 6)}
            for a in DEFAULT_ARMS
        }


def get_rerank_bandit(project_root: Path) -> RerankBandit:
    global _bandit_singleton
    if _bandit_singleton is None:
        _bandit_singleton = RerankBandit(project_root / "rl_rerank_bandit_state.json")
    return _bandit_singleton
