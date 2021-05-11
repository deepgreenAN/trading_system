import gym

def make_concrete_env():
    one_concrete_env = gym.make("Pendulum-v0")
    return one_concrete_env