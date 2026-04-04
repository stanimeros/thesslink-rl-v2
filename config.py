"""Single source of truth for the active environment version.

Change ENV_VERSION here and it will propagate everywhere:
  train.sh, visualize.py, smoke_test.py, visualization.py
"""

ENV_VERSION = 1  # 0 = grid obs (313 features), 1 = symbolic obs (23 features)

# --- Derived (do not edit) ------------------------------------------------

if ENV_VERSION == 1:
    from thesslink_rl.v1 import ENV_TAG, GridNegotiationEnv
else:
    from thesslink_rl.v0 import ENV_TAG, GridNegotiationEnv

ENV_CONFIG = "thesslink_v1" if ENV_VERSION == 1 else "thesslink"
