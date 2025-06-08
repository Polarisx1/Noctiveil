# Noctiveil

## Game mode tuning

The bot automatically adjusts a few parameters based on the number of teammates
in the match. When the first game packet is received, it checks whether the
match is a 1v1, 2v2 or 3v3 and updates `tick_skip`, `beta` and whether hardcoded
kickoffs are used. These values can be customised in `bot.cfg` under the
`[Tuning]` section. Defaults are shown below:

```
[Tuning]
tick_skip_1v1 = 8
tick_skip_2v2 = 6
tick_skip_3v3 = 6
beta_1v1 = 1.0
beta_2v2 = 0.9
beta_3v3 = 0.8
kickoff_1v1 = true
kickoff_2v2 = true
kickoff_3v3 = false
```

Edit these values to fine tune behaviour for each game mode.
