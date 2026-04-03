# ThessLink RL v2 -- Multi-Agent Grid Negotiation for EPyMARL

Multi-Agent Reinforcement Learning environment where two agents **negotiate** over Points of Interest (POIs) on a 10x10 grid, then **navigate** to the agreed target. Designed to plug into [EPyMARL](https://github.com/uoe-agents/epymarl) for training with algorithms like QMIX, MAPPO, VDN, IQL, and more.

## Architecture

| File | Purpose |
|---|---|
| `thesslink_rl/environment.py` | Core env -- 10x10 grid, obstacles, POIs, comms |
| `thesslink_rl/evaluation.py` | POI preference scoring (energy, privacy) |
| `thesslink_rl/negotiation.py` | Heuristic negotiation -- agents exchange scores, pick best POI |
| `thesslink_rl/gym_wrapper.py` | Gymnasium multi-agent wrapper for EPyMARL compatibility |
| `thesslink_rl/visualization.py` | Grid rendering, training curves, episode replay GIF |
| `thesslink_rl/models/` | Agent type configs (YAML): human, taxi, drone |
| `epymarl_config/thesslink.yaml` | Ready-to-use EPyMARL env config |

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

### Environment Phases

1. **Negotiation** (automatic at reset): Each agent scores all POIs based on its energy/privacy config, exchanges scores over several rounds, then the POI maximising the product of both agents' scores is chosen.

2. **Navigation** (learned by EPyMARL): Agents receive observations and take discrete movement actions (stay, up, down, left, right) to reach the agreed POI. This is the phase controlled by the RL algorithm.

### Observation Space

Each agent observes a flat vector of size **303**:
- Grid `(3 x 10 x 10 = 300)` float32 -- channels: obstacles, POIs, self position
- Comm `(3,)` float32 -- the peer agent's latest POI score broadcast

### Action Space

`Discrete(5)` per agent: stay, up, down, left, right.

### Rewards

- **Common reward mode** (default): both agents receive the Golden Mean reward (`Score_A x Score_B`) when any agent reaches the target POI.
- **Individual reward mode** (`common_reward=False`): each agent receives its own Golden Mean reward.

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
from thesslink_rl.visualization import render_grid
from thesslink_rl.environment import GridNegotiationEnv

env = GridNegotiationEnv(seed=42)
env.reset()
render_grid(env, title="Initial State")
```
