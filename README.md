# ThessLink RL - Multi-Agent Grid Negotiation for EPyMARL

Two agents **negotiate** a meeting spot on a 10×10 grid, then **navigate** there. Both phases are learned with RL. Works with [EPyMARL](https://github.com/uoe-agents/epymarl) (QMIX, MAPPO, IQL, VDN, etc.).

## The task

Each episode: two agents spawn on a map with obstacles and **three POIs** (candidate meeting places). From YAML profiles they each prefer some POIs over others—**easier to reach** and/or **better for privacy** from where they started. They take turns suggesting or accepting until they **agree on one POI**, then both **walk to it**. The run ends successfully only when **both** have reached the agreed cell (whoever arrives first waits).

## Environment versions

| Version | Observation | Notes |
|--------|-------------|--------|
| **v0** | Full grid flattened (311 values) | Both agents suggest every step; agreement when they pick the **same** POI on the **same** step. |
| **v1** | Compact vector (19 values) | Turn-based suggest / accept; “GPS” toward target + obstacle sense; no full grid in the observation. |
| **v2** | Same as v1 | Same rules as v1, **plus** extra reward shaping for negotiation and navigation (the default setup in this repo). |

Local scripts use **`config.py` → `ENV_VERSION`** (default **2**). For EPyMARL, pick the matching config name, e.g. **`thesslink_v2`**.

## Environment (v2)

| Aspect | Description |
|--------|-------------|
| Grid | $10 \times 10$, **10** random obstacles, **3** POIs, **2** agents |
| Phases | **Negotiation** (turn-based) then **Navigation** until both stand on the agreed POI |
| Agreement | Active agent suggests a POI or **accepts** the peer’s last suggestion; on accept, navigation begins |
| Agent models | YAML files in `thesslink_rl/models/` (energy model + privacy emphasis) |

## Observation (v2)

Flat vector length **19** (same layout in both phases; values in roughly $[0,1]$ unless noted).

| Block | Size | Content |
|-------|------|---------|
| Phase | 1 | $0$ = negotiation, $1$ = navigation |
| POI scores | 3 | This agent’s preference score for POI $0,1,2$ (from evaluation below) |
| Peer action | 4 | One-hot: no suggestion yet, or peer suggested POI $0,1,2$ |
| Agreed POI | 3 | One-hot of locked-in POI (zeros until agreement) |
| Self position | 2 | Row and column, normalised by grid extent |
| Relative offset | 2 | Toward agreed POI in navigation (zeros in negotiation) |
| Lidar | 4 | Normalised distance to nearest obstacle N, S, E, W |

## Actions (v2)

| ID | Meaning |
|----|---------|
| 0 | Stay |
| 1 | Up |
| 2 | Down |
| 3 | Left |
| 4 | Right |
| 5 | Suggest POI 0 |
| 6 | Suggest POI 1 |
| 7 | Suggest POI 2 |
| 8 | Accept peer’s last suggestion |

In **negotiation**, only the active agent may use actions **5–8**; the other is restricted to a no-op. In **navigation**, both use **0–4**.

## Rewards (v2)

| Phase | What happens | Reward idea |
|-------|----------------|-------------|
| Negotiation | Suggest your best POI, push back on bad offers, or accept a fair one | Small shaping bonuses |
| Negotiation | Agreement reached | $+10 \times \text{quality}$ for everyone (quality from evaluation below) |
| Navigation | Each step | Potential-based move toward target, minus small step cost |
| Navigation | One agent reaches POI | $+10 \times \text{quality}$ for that agent |
| Navigation | **Both** at POI | Extra $+50 \times \text{quality}$ for everyone |

## Evaluation (example: Drone)

[`thesslink_rl/models/drone.yaml`](thesslink_rl/models/drone.yaml) sets **privacy emphasis** $\alpha = 0.4$ and a **linear** energy model. For each POI $k$ the code uses **travel cost** from the agent’s **current** cell (min–max across the three POIs; lower cost ⇒ higher $\tilde{E}_k$) and **privacy** as spawn-to-POI BFS distance divided by the **maximum** BFS distance reachable from spawn on the map (not min–max across POIs), so $\tilde{P}_k\in[0,1]$ without forcing the nearest POI to privacy $0$ unless it is almost at spawn.

$$
s_k \;=\; (1-\alpha)\,\tilde{E}_k + \alpha\,\tilde{P}_k\,.
$$

After agreement on POI $k^\star$, **negotiation quality** compares how good that choice is for **both** agents to the best possible common POI. Let $s_k^{(a)}$ be agent $a$’s score for POI $k$. The golden-mean vector is $g_k = \prod_a s_k^{(a)}$, and

$$
\text{quality} \;=\; \frac{g_{k^\star}}{\max_\ell g_\ell} \;\in\; [0,1]\,.
$$

That scalar scales the shared agreement and navigation bonuses in the reward table above.

![Agent evaluation heatmaps: same map, different POI scores and preference fields for two agent types](plots/v2/eval_heatmaps.png)

## Plots and replays

```bash
python visualize.py
```

Outputs under `plots/<env_tag>/` (e.g. `plots/v2/`):

![Training curves — all algorithms](plots/v2/training_curves-all.png)

**MAPPO** — episode replay (agreement **100%**, golden-mean **93.3%**, reach **100%**)

![Episode replay — MAPPO](plots/v2/episode_replay-mappo.gif)

**IQL** — episode replay (agreement **100%**, golden-mean **96.9%**, reach **100%**)

![Episode replay — IQL](plots/v2/episode_replay-iql.gif)

## Algorithms

EPyMARL supports value-decomposition methods (QMIX, VDN, …), actor–critic and policy-gradient variants (MAPPO, IPPO, COMA, …), and independent learners (IQL, IA2C, …). **`train.sh`** passes **`common_reward=False`** for every algorithm so learning uses **per-agent** rewards from the v2 wrapper (you can override when invoking `main.py` yourself).

## Rendering (optional)

```python
from config import GridNegotiationEnv
from thesslink_rl.visualization import render_grid, capture_frame

env = GridNegotiationEnv(seed=42)
env.reset()
render_grid(env, title="Initial State")
```
