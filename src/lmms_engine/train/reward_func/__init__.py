from .grpo_reward import accuracy_reward, format_reward

REWARD_FUNC_REGISTRY = {"accuracy": accuracy_reward, "format": format_reward}

__all__ = ["REWARD_FUNC_REGISTRY"]
