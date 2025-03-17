import operator
import math
import random
import numpy as np
import multiprocessing
from deap import base, creator, gp, tools, algorithms
import cv2
import matplotlib.pyplot as plt
import time
from env import EnvBreakoutEasy
from tqdm import tqdm
from functools import partial
import pickle
import os
import datetime
import csv
import glob

# Setup same environment and primitives as in training
env = EnvBreakoutEasy(render_mode=None)
input_size = env.observation_space[0]
n_actions = env.action_space.n

pset = gp.PrimitiveSetTyped("MAIN", [float] * input_size, int)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(math.cos, [float], float)

def if_then_else(condition, out1, out2):
    return out1 if condition else out2
pset.addPrimitive(if_then_else, [bool, int, int], int)

def lt(a, b):
    return a < b
pset.addPrimitive(lt, [float, float], bool)

def gt(a, b):
    return a > b
pset.addPrimitive(gt, [float, float], bool)

pset.addEphemeralConstant("const", partial(random.uniform, 0, 255), float)

pset.addTerminal(0, int)
pset.addTerminal(1, int)
pset.addTerminal(2, int)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def load_best_individual(filepath=None):
    """
    Load the best individual from a specified file or find the most recent one.
    """
    if filepath is None:
        # Find the most recent best individual file
        files = glob.glob("gp_checkpoints/best_individual_*.pkl") + glob.glob("gp_results/best_individual_*.pkl")
        if not files:
            print("No saved individuals found.")
            return None
        
        # Sort by modification time (most recent first)
        filepath = max(files, key=os.path.getmtime)
        print(f"Using most recent individual: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            best_individual = pickle.load(f)
            print(f"Successfully loaded individual from {filepath}")
            return best_individual
    except Exception as e:
        print(f"Error loading individual: {e}")
        return None

def evaluate_individual(individual, num_games=10):
    """
    Evaluate an individual over multiple games and return statistics.
    """
    func = toolbox.compile(expr=individual)
    rewards = []
    
    print(f"Evaluating individual over {num_games} games...")
    env_eval = EnvBreakoutEasy(render_mode=None)
    
    for i in range(num_games):
        state = env_eval.reset()
        game_reward = 0
        done = False
        
        while not done:
            obs = [float(x) for x in state[:input_size]]
            action = func(*obs)
            action = int(action) % n_actions
            state, reward, done, _ = env_eval.step(action)
            game_reward += reward
        
        rewards.append(game_reward)
        print(f"Game {i+1}: Score = {game_reward}")
    
    return {
        "min": min(rewards),
        "max": max(rewards),
        "avg": sum(rewards) / len(rewards),
        "std": np.std(rewards),
        "all_scores": rewards
    }

def display_game(individual, delay=50, record=False, output_path=None):
    """
    Display a game played by the given individual.
    
    Args:
        individual: The individual to evaluate
        delay: Delay between frames in ms (lower = faster)
        record: Whether to record the game as a video
        output_path: Path to save the video if recording
    """
    env_vis = EnvBreakoutEasy(render_mode='human')
    state = env_vis.reset()
    func = toolbox.compile(expr=individual)
    
    frames = []
    total_reward = 0
    done = False
    
    # Video writer setup
    if record:
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"replay_{timestamp}.mp4"
        
        frame = env_vis.Pixel_Obs()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 1000/delay, (w, h))
    
    print("Starting game replay... Press 'q' to quit.")
    
    while not done:
        obs = [float(x) for x in state[:input_size]]
        action = func(*obs)
        action = int(action) % n_actions
        state, reward, done, _ = env_vis.step(action)
        total_reward += reward
        
        env_vis.render()
        frame = env_vis.Pixel_Obs()
        
        if record:
            out.write(frame)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Game Replay", frame_bgr)
        
        # Display score on screen
        cv2.putText(
            frame_bgr, f"Score: {total_reward}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    print(f"Game finished with score: {total_reward}")
    
    if record:
        out.release()
        print(f"Video saved to {output_path}")
    
    cv2.destroyAllWindows()
    if hasattr(env_vis, "close"):
        env_vis.close()
    
    return total_reward

def main():
    print("=" * 50)
    print("Game Replay Tool")
    print("=" * 50)
    
    # List available saved individuals
    checkpoints = glob.glob("gp_checkpoints/best_individual_*.pkl") + glob.glob("gp_results/best_individual_*.pkl")
    
    if checkpoints:
        print("\nAvailable saved individuals:")
        for i, path in enumerate(checkpoints):
            print(f"[{i}] {path}")
        
        choice = input("\nEnter number to select individual (or press Enter for most recent): ")
        
        if choice.strip():
            try:
                filepath = checkpoints[int(choice)]
            except (ValueError, IndexError):
                print("Invalid selection. Using most recent individual.")
                filepath = None
        else:
            filepath = None
    else:
        print("No saved individuals found.")
        return
    
    # Load the individual
    individual = load_best_individual(filepath)
    if individual is None:
        return
    
    while True:
        print("\nOptions:")
        print("1. Play game (visualize)")
        print("2. Record game as video")
        print("3. Evaluate performance (run multiple games)")
        print("4. Show individual's tree structure")
        print("5. Exit")
        
        option = input("Select option: ")
        
        if option == "1":
            speed = input("Playback speed (ms delay, default=50, lower=faster): ")
            delay = int(speed) if speed.strip().isdigit() else 50
            display_game(individual, delay=delay)
        
        elif option == "2":
            speed = input("Playback speed (ms delay, default=50, lower=faster): ")
            delay = int(speed) if speed.strip().isdigit() else 50
            path = input("Output path (leave empty for default): ")
            display_game(individual, delay=delay, record=True, output_path=path if path.strip() else None)
        
        elif option == "3":
            num_games = input("Number of games to evaluate (default=10): ")
            num_games = int(num_games) if num_games.strip().isdigit() else 10
            stats = evaluate_individual(individual, num_games)
            
            print("\nPerformance Statistics:")
            print(f"Min score: {stats['min']}")
            print(f"Max score: {stats['max']}")
            print(f"Average score: {stats['avg']:.2f}")
            print(f"Standard deviation: {stats['std']:.2f}")
            
            # Plot histogram of scores
            plt.figure(figsize=(10, 6))
            plt.hist(stats['all_scores'], bins=10, color='skyblue', edgecolor='black')
            plt.title('Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        elif option == "4":
            print("\nIndividual's tree structure:")
            print(individual)
            print(f"Tree size: {len(individual)} nodes")
            
            # Visualize tree
            nodes, edges, labels = gp.graph(individual)
            
            import networkx as nx
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
            
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(g, pos, node_size=900, node_color="lightblue")
            nx.draw_networkx_edges(g, pos)
            nx.draw_networkx_labels(g, pos, labels)
            plt.axis("off")
            plt.title("Individual's Tree Structure")
            plt.tight_layout()
            plt.show()
        
        elif option == "5":
            break
        
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()