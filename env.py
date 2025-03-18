import time
import gym
import numpy as np
import pickle
import threading
from gym.spaces import Discrete, Box
from gym import Wrapper
from pynput import keyboard
import os
import random
import ale_py

# Global variable to store the last key released
last_key_released = None
key_lock = threading.Lock()

#only catches space bar
def on_press(key):
    global last_key_released
    with key_lock:
        try:
            if key.char == ' ':
                last_key_released = key.char
        except AttributeError:
            if key.name == 'space':
                last_key_released = key.name

def on_release(key):
    global last_key_released
    with key_lock:
        try:
            last_key_released = key.char
        except AttributeError:
            last_key_released = key.name

# Create a keyboard listener that only uses on_release events.
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

import gym
import random
import numpy as np
from gym import Wrapper
from gym.spaces import Discrete, Box

class EnvBreakout(Wrapper):
    def __init__(self, render_mode=None, clipping=True, negative_reward=True):
        # Create the underlying Atari Breakout env
        self.env = gym.make('Breakout-ram-v4',
                            full_action_space=False,
                            frameskip=4,
                            render_mode=render_mode)
        super().__init__(self.env)
        self.clipping = clipping
        self.negative_reward = negative_reward

        # Reduced 3-action space:
        #   0 -> NO-OP
        #   1 -> LEFT
        #   2 -> RIGHT
        # Mapped to underlying ALE actions: NO-OP=0, LEFT=2, RIGHT=3
        self.action_space = Discrete(3)
        self.action_mapper = {0: 0, 1: 2, 2: 3}

        # Same shape, but we use raw RAM for illustration
        self.observation_space = Box(low=0, high=255, shape=(128,), dtype=np.uint8)
        self.metadata = self.env.metadata

        # Track how many lives are left
        self.num_lives = 5

        # A flag that tells us whether we need to press FIRE after losing a life
        self._need_fire = False

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=random.randint(0, 9999))
        self.num_lives = 5

        # Press FIRE to start the game
        obs, real_r, done, truncated, info = self.env.step(1)
        return self.RAM_Obs(obs), info

    def step(self, action, return_real_reward=False):
        """
        Performs the agent's intended action (left/right/no-op).
        If a life was lost (and there are still lives left), sets a flag that
        we need to FIRE on the *next* call to 'step_fire_if_needed'.
        """

        mapped_action = self.action_mapper[action]
        obs, real_reward, done, truncated, info = self.env.step(mapped_action)

        # Possibly clip the environment's reward
        if self.clipping:
            reward = np.clip(real_reward, -1, 1)
        else:
            reward = real_reward

        # Check if a life was lost
        remaining_lives = info.get('lives', self.num_lives)
        lost_life = (remaining_lives < self.num_lives)
        self.num_lives = remaining_lives

        if lost_life and remaining_lives > 0 and not done and not truncated:
            # Apply the negative penalty (if requested)
            if self.negative_reward:
                reward += -1
            # Mark that we must FIRE on the next call to step_fire_if_needed()
            self._need_fire = True

        # If no more lives, or the ALE says game is over
        if remaining_lives == 0 or done or truncated:
            done = True

        # Return real_reward as well if asked
        if return_real_reward:
            return self.RAM_Obs(obs), reward, done, truncated, info, real_reward
        else:
            return self.RAM_Obs(obs), reward, done, truncated, info

    def step_fire_if_needed(self, return_real_reward=False):
        """
        If a life was lost in the previous transition (and we still have lives),
        do the forced 'FIRE' step here, returning its new observation/reward.
        Otherwise, return None.
        """
        if self._need_fire:
            self._need_fire = False
            obs, real_reward, done, truncated, info = self.env.step(1)
            if self.clipping:
                reward = np.clip(real_reward, -1, 1)
            else:
                reward = real_reward

            if return_real_reward:
                return self.RAM_Obs(obs), reward, done, truncated, info, real_reward
            else:
                return self.RAM_Obs(obs), reward, done, truncated, info

        return None

    def close(self):
        self.env.close()

    @staticmethod
    def RAM_Obs(obs):
        # If you ever want to filter or reshape the RAM, do it here.
        return obs


def collect_human_imitation_dataset(env, num_episodes=10, max_steps=5000, dataset_file='human_imitation_dataset.pkl'):
    """
    Collects human demonstration data using asynchronous keyboard input on key release.

    Controls (release keys):
      - 'q' -> LEFT (action index 1)
      - 'd' -> RIGHT (action index 2)
      - Any other key -> NO-OP (action index 0)
      - 'esc' -> Exit the current episode early

    Each transition is stored as a tuple: (state, action, reward, next_state, done)
    """
    global last_key_released
    dataset = []  # List to hold episodes

    print("== Human Demonstration Data Collection ==")
    print("Controls (release keys):")
    print("  q -> LEFT")
    print("  d -> RIGHT")
    print("  any other key -> NO-OP")
    print("  esc -> quit current episode\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        episode_data = []
        print(f"Starting episode {episode+1}/{num_episodes}")

        while not done and step_count < max_steps:
            env.render()
            action = None

            # Wait for a key release event
            while action is None:
                with key_lock:
                    key = last_key_released
                    if key is not None:
                        # Once a key is registered, reset the global variable
                        last_key = key
                        last_key_released = None
                        if last_key == 'q':
                            action = 2  # LEFT
                        elif last_key == 'd':
                            action = 1  # RIGHT
                        elif last_key == 'esc':
                            print("Exiting episode early.")
                            done = True
                            break
                        else:
                            action = 0  # NO-OP
                time.sleep(0.01)  # Prevent high CPU usage

            if done:
                break

            # Step the environment with the chosen action
            next_state, reward, done, truncated, info = env.step(action)
            episode_data.append((state, action, reward, next_state, done))
            state = next_state
            step_count += 1

            if done or truncated:
                print("Episode finished.")
                break

        dataset.append(episode_data)
        print(f"Episode {episode+1} recorded with {step_count} steps.\n")

        # Save the collected dataset to a file.
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_file}")

def concatenate_human_dataset(dataset_folder = ""):
    datasets_files = os.listdir(dataset_folder)
    final_dataset = []
    for k in datasets_files:
        with open(dataset_folder + k, 'rb') as f:
            dataset = pickle.load(f)
            final_dataset.extend(dataset)
    return final_dataset

if __name__ == '__main__':
    env = EnvBreakout(render_mode='human', clipping=True)
    collect_human_imitation_dataset(env, num_episodes=50, dataset_file='human_imitation_dataset.pkl')
    env.close()
    

