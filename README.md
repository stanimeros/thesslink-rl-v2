# ThessLink RL -- Multi-Agent Grid Negotiation for EPyMARL

Multi-Agent Reinforcement Learning environment where two agents **negotiate** over Points of Interest (POIs) on a 10x10 grid, then **navigate** to the agreed target. Both phases are **RL-trained** -- agents learn to negotiate through suggest/accept actions and to navigate with movement actions. Designed to plug into [EPyMARL](https://github.com/uoe-agents/epymarl) for training with algorithms like QMIX, MAPPO, VDN, IQL, and more.

## Architecture

| File | Purpose |
|---|---|
| `thesslink_rl/environment.py` | Core env -- 10x10 grid, obstacles, POIs, negotiation + navigation phases |
| `thesslink_rl/evaluation.py` | POI preference scoring (energy, privacy) and Golden Mean reward |
| `thesslink_rl/gym_wrapper.py` | Gymnasium multi-agent wrapper for EPyMARL compatibility |
| `thesslink_rl/visualization.py` | Grid rendering, training curves, episode replay GIF |
| `thesslink_rl/models/` | Agent type configs (YAML): human, taxi, drone |
| `epymarl/src/config/envs/thesslink.yaml` | EPyMARL env config |

## Quick Start

### 1. Install the environment

```bash
pip install -e .
```

### 2. Clone EPyMARL

```bash
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl
pip install -r requirements.txt
```

### 3. Run an algorithm

Copy ThessLink env configs into EPyMARL (add new `thesslink_*.yaml` files under `epymarl_config/envs/` as needed):

```bash
cp ../epymarl_config/envs/*.yaml src/config/envs/
```

Then run any algorithm:

```bash
# QMIX (common reward)
python src/main.py --config=qmix --env-config=thesslink

# MAPPO (individual rewards)
python src/main.py --config=mappo --env-config=thesslink with common_reward=False

# VDN
python src/main.py --config=vdn --env-config=thesslink

# IQL
python src/main.py --config=iql --env-config=thesslink

# IA2C
python src/main.py --config=ia2c --env-config=thesslink with common_reward=False
```

Or use the gymma config directly without copying:

```bash
python src/main.py --config=qmix --env-config=gymma \
  with env_args.time_limit=60 env_args.key="thesslink_rl:thesslink/GridNegotiation-v0"
```

## Environment

### Grid

- **10x10** grid with **10** random obstacles and **3** Points of Interest (POIs)
- **2** agents spawn at random free cells each episode
- Each agent has a YAML config defining its energy model and privacy preference

### Episode Flow

Each episode has two RL-controlled phases sharing a **60-step budget**:

1. **Negotiation** (learned): Agents observe their own POI preference scores, the peer's last suggestion, the grid layout, and their position. Each step, they choose a suggest action (suggest POI 0, 1, or 2). When both agents suggest the **same POI on the same step**, they agree and the episode transitions to navigation.

2. **Navigation** (learned): Agents continue to see the full grid and negotiation state. They take movement actions (stay, up, down, left, right) to reach the agreed POI.

Both phases count against the same 60-step limit, so agents are incentivized to negotiate quickly to leave enough time for navigation.

### Action Space

`Discrete(8)` per agent with **phase-dependent action masking**:

| Action | Meaning | Valid During |
|--------|---------|-------------|
| 0 | Stay | Navigation |
| 1 | Up | Navigation |
| 2 | Down | Navigation |
| 3 | Left | Navigation |
| 4 | Right | Navigation |
| 5 | Suggest POI 0 | Negotiation |
| 6 | Suggest POI 1 | Negotiation |
| 7 | Suggest POI 2 | Negotiation |

EPyMARL uses `get_avail_actions()` to mask invalid actions per phase.

### Observation Space

**Unified** flat vector of size **311**, identical layout in both phases:

| Segment | Size | Content |
|---------|------|---------|
| Phase flag | 1 | 0.0 = negotiation, 1.0 = navigation |
| POI scores | 3 | Agent's preference score for each POI (energy + privacy) |
| Peer action | 4 | One-hot of peer's last suggestion, or "no action yet" |
| Agreed POI | 3 | One-hot of agreed POI (all zeros until agreement) |
| Grid | 300 | 3-channel 10x10 grid: obstacles, POI locations, self position |

The agent always sees the full picture -- its position on the map, the negotiation state, and which phase it's in -- regardless of the current phase.

### Agreement Protocol

1. Episode starts in `negotiation` phase
2. Each step, both agents simultaneously choose a suggest action (5, 6, or 7)
3. When both agents suggest the **same POI on the same step**, they agree
4. The env switches to `navigation` phase with the agreed POI as target
5. If agents spend all 60 steps negotiating, the episode truncates with zero reward

### Rewards

| Event | Reward | Purpose |
|-------|--------|---------|
| Agreement on POI | `score_a * score_b` | Learn to pick mutually beneficial POIs |
| Each nav step closer to POI | `+0.01 * distance_reduced` | Learn to navigate efficiently |
| Each nav step away from POI | `-0.01 * distance_increased` | Penalize wasted moves |
| Reaching the agreed POI | `score_a * score_b` | Terminal bonus for completing the task |
| Episode timeout | 0 | No additional penalty |

Distance is measured using BFS (shortest path respecting obstacles).

### POI Scoring Formula

```
score = (1 - p) * energy + p * privacy       where p = privacy_emphasis
```

- **Energy** -- how cheap is it to reach the POI from the current position? (BFS distance, linear or exponential cost model). Normalized to [0, 1].
- **Privacy** -- does visiting the POI conceal the agent's spawn location? A POI far from spawn = high privacy. Normalized to [0, 1].

## Agent Configs

Define agent types as YAML files in `thesslink_rl/models/`:

```yaml
# thesslink_rl/models/drone.yaml
name: Drone
privacy_emphasis: 1.0
energy_model: linear
energy_per_step: 1.0
energy_exponential_gamma: 0.12
```

## Available EPyMARL Algorithms

| Algorithm | Individual Rewards | Common Rewards |
|---|---|---|
| QMIX | - | Yes |
| VDN | - | Yes |
| COMA | - | Yes |
| MAPPO | Yes | Yes |
| IPPO | Yes | Yes |
| MAA2C | Yes | Yes |
| IA2C | Yes | Yes |
| IQL | Yes | Yes |
| PAC | Yes | Yes |

## Visualization

The `thesslink_rl.visualization` module can still be used for rendering and debugging:

```python
from thesslink_rl.visualization import render_grid, capture_frame
from thesslink_rl.environment import GridNegotiationEnv

env = GridNegotiationEnv(seed=42)
env.reset()
render_grid(env, title="Initial State")
```
