"""Single source of truth for the active environment version.

Change ENV_VERSION here and it will propagate everywhere:
  train.sh, visualize.py, smoke_test.py, visualization.py
"""

ENV_VERSION = 2  # 0 = grid obs, 1 = symbolic obs, 2 = symbolic obs + reward shaping

# --- Derived (do not edit) ------------------------------------------------

if ENV_VERSION == 2:
    from thesslink_rl.v2 import ENV_TAG, GridNegotiationEnv
elif ENV_VERSION == 1:
    from thesslink_rl.v1 import ENV_TAG, GridNegotiationEnv
else:
    from thesslink_rl.v0 import ENV_TAG, GridNegotiationEnv

_ENV_CONFIG_MAP = {0: "thesslink", 1: "thesslink_v1", 2: "thesslink_v2"}
ENV_CONFIG = _ENV_CONFIG_MAP.get(ENV_VERSION, "thesslink")
