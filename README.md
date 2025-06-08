# Noctiveil

CarbonXBot is a Rocket League bot built using the [RLBot](https://github.com/RLBot/RLBot) framework. It mixes ideas from the well–known Nexto bot with custom utilities from this repository. The end goal is to create a strong rival to Nexto that is easy to tune and experiment with.

## Installation

1. Install Python 3.9 or newer.
2. Install the required packages:
   ```bash
   pip install rlbot numpy torch rlgym_compat
   ```
3. Clone this repository and open a terminal in the project folder.

## Launching the Bot

RLBot looks for a configuration file that tells it which script to run and how the bot should appear. Run the bot with:

```bash
python -m rlbot ./bot.cfg
```

Alternatively, open RLBotGUI and select `bot.cfg` to add the bot to a match.

## Configuration

### bot.cfg

- `looks_config` &ndash; path to the appearance configuration (`appearance.cfg`).
- `python_file` &ndash; entry point for the bot (`bot.py`).
- `name` &ndash; the bot name shown in game.
- `maximum_tick_rate_preference` &ndash; desired tick rate (120 by default).
- The `[Details]` section contains optional metadata.

### appearance.cfg

This file defines the car loadout and paints for blue and orange teams. You can edit it directly or use RLBotGUI's appearance editor to tweak wheels, boosts and other cosmetics.

### bot.py parameters

Inside `bot.py` the `Nexto` class exposes several tuning variables:

```python
self.tick_skip = 8
# Beta controls randomness:
# 1=best action, 0.5=sampling from probability, 0=random, -1=worst action
self.beta = beta
```

Adjust `tick_skip` to change how often the bot chooses a new action and modify `beta` to make the bot more deterministic or more random.

## Nexto's Rival Goal

The long‑term aim of CarbonXBot is to approach or surpass the performance of Nexto. Experiment with the `beta` setting and other parameters to tune the bot's aggressiveness and reliability. Future work will involve refining these values and improving the underlying model to compete directly with Nexto.

