# ThessLink RL v2 -- Multi-Agent Grid Negotiation for EPyMARL

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

Copy the env config into EPyMARL:

```bash
cp ../epymarl_config/thesslink.yaml src/config/envs/thesslink.yaml
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

## How It Works

### Episode Flow

Each episode has two RL-controlled phases sharing a **60-step budget**:

1. **Negotiation** (learned): Agents observe their own POI preference scores and the peer's last suggestion. Each step, they choose a suggest action (suggest POI 0, 1, or 2). When both agents suggest the **same POI on the same step**, they agree and the episode transitions to navigation.

2. **Navigation** (learned): Agents observe the full grid (obstacles, POIs, own position) and take movement actions (stay, up, down, left, right) to reach the agreed POI.

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

Flat vector of size **303** (zero-padded when needed):

**During negotiation** (7 floats, padded to 303):
- Own POI scores `(3,)` -- computed from the agent's energy/privacy config
- Peer's last action one-hot `(4,)` -- [suggest_0, suggest_1, suggest_2, no_action_yet]

**During navigation** (303 floats):
- Grid `(3 x 10 x 10 = 300)` float32 -- channels: obstacles, POIs, self position
- Comm `(3,)` float32 -- the peer agent's latest broadcast

RNN-based algorithms (QMIX, MAPPO, etc.) handle the temporal history of suggest actions during negotiation.

### Agreement Protocol

1. Episode starts in `negotiation` phase
2. Each step, both agents simultaneously choose a suggest action (5, 6, or 7)
3. When both agents suggest the **same POI on the same step**, they agree
4. The env switches to `navigation` phase with the agreed POI as target
5. If agents spend all 60 steps negotiating, the episode truncates with zero reward

### Rewards

- **During negotiation**: 0 reward per step
- **On reaching agreed POI**: Golden Mean reward (`Score_A x Score_B`) for both agents
- **Truncated** (no agreement or POI not reached): 0 reward

### POI Scoring Formula

```
score = (1 - p) * energy + p * privacy       where p = privacy_emphasis
```

- **Energy** -- how cheap is it to reach the POI? (linear or exponential model)
- **Privacy** -- does visiting the POI reveal the agent's spawn location?

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
