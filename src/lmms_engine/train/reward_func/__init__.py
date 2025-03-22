try:
    from .grpo_reward import accuracy_reward, format_reward
except Exception as e:
    print(f"Error: {e} when importing reward functions")
    accuracy_reward = None
    format_reward = None

REWARD_FUNC_REGISTRY = {"accuracy": accuracy_reward, "format": format_reward}

__all__ = ["REWARD_FUNC_REGISTRY"]
