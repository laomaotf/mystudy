import gym
from gym.utils.play import play,PlayPlot
import pygame

def testbed_random(render_mode="human"):
    env = gym.make("LunarLander-v2",render_mode=render_mode)
    observation, info = env.reset(seed=42)
    print(env.action_space)
    for _ in range(1000):
        #action = policy(observation)  # User-defined policy function
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    
def play_game(game_name="LunarLander-v2"):
    mapping = {
        (pygame.K_LEFT,):1, (pygame.K_RIGHT,):3,(pygame.K_DOWN,):2
    }
    def callback(obs_t, obs_tp1, action, reward, done,truncated,info):
        return [reward,]
    plotter = PlayPlot(callback, 30 * 5, ["reward"])
    play(gym.make(game_name,render_mode="rgb_array"),
         keys_to_action=mapping,callback=plotter.callback)
 
        

if __name__ == "__main__":
    play_game()