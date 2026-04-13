"""Single source of truth for the active environment version.

There is **no** default: set ``THESSLINK_ENV_VERSION`` (``"0"``..``"3"``) in the
environment before importing this module. ``train.sh`` and ``visualize.py`` set it
after you pick a version (CLI flag or prompt).
"""

import os

if "THESSLINK_ENV_VERSION" not in os.environ:
    raise RuntimeError(
        "THESSLINK_ENV_VERSION is not set (expected 0–3). "
        "Use ./train.sh --env N …, python visualize.py --env N, or export THESSLINK_ENV_VERSION=N.",
    )
ENV_VERSION = int(os.environ["THESSLINK_ENV_VERSION"])
if ENV_VERSION not in (0, 1, 2, 3):
    raise ValueError(
        f"THESSLINK_ENV_VERSION must be 0–3, got {ENV_VERSION!r}",
    )

# --- Derived (do not edit) ------------------------------------------------

if ENV_VERSION == 3:
    from thesslink_rl.v3 import ENV_TAG, GridNegotiationEnv
elif ENV_VERSION == 2:
    from thesslink_rl.v2 import ENV_TAG, GridNegotiationEnv
elif ENV_VERSION == 1:
    from thesslink_rl.v1 import ENV_TAG, GridNegotiationEnv
else:
    from thesslink_rl.v0 import ENV_TAG, GridNegotiationEnv

_ENV_CONFIG_MAP = {
    0: "thesslink",
    1: "thesslink_v1",
    2: "thesslink_v2",
    3: "thesslink_v3",
}
ENV_CONFIG = _ENV_CONFIG_MAP.get(ENV_VERSION, "thesslink")
