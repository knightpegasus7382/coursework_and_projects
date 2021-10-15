from gym.envs.registration import register

register(
    'chakra-v0',
    entry_point='chakra.envs:chakra',
    #max_episode_steps=40,
)
