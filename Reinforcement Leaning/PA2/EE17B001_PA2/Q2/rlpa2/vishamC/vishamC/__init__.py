from gym.envs.registration import register

register(
    'vishamC-v0',
    entry_point='vishamC.envs:vishamC',
    #max_episode_steps=40,
)
