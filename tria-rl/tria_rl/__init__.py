from gym.envs.registration import register

register(
    id="tria_rl/TriaClimate-v0",
    entry_point="tria_rl.envs:TriaClimateEnv",
)
