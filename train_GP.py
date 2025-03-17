import operator
import math
import random
import numpy as np
import multiprocessing
from deap import base, creator, gp, tools, algorithms
import cv2
import matplotlib.pyplot as plt
import time
from env import EnvBreakout
from tqdm import tqdm
from functools import partial
import pickle
import os
import datetime
import csv

env = EnvBreakout(render_mode=None,negative_reward=False, featureTuning=True)
input_size = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"Observation space: {input_size}, Action space: {n_actions}")

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

NB_RUNS = 10
SIZE_PENALTY_FACTOR = 0.035

def eval_individual(individual):
    func = toolbox.compile(expr=individual)
    total_reward = 0
    for _ in range(NB_RUNS):
        state, _  = env.reset()
        run_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            obs = [float(x) for x in list(state)]
            action = func(*obs)
            action = int(action) % n_actions
            state, reward, done, truncated, _ = env.step(action)
            run_reward += reward
        
        total_reward += run_reward
    size_penalty = SIZE_PENALTY_FACTOR * len(individual)
    return ((total_reward / NB_RUNS) - size_penalty,)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def display_game(individual): 
    """
    Affiche une partie (simulation) du meilleur individu sur un environnement en mode 'human'.
    """
    env_vis = EnvBreakout(render_mode='human', featureTuning=True, negative_reward=False)
    state, _  = env_vis.reset()
    best_func = toolbox.compile(expr=individual)
    plt.ion()
    done = False
    truncated = False
    while not (done or truncated):
        obs = [float(x) for x in list(state)]
        action = best_func(*obs)
        action = int(action) % n_actions
        state, _, done, truncated, _ = env_vis.step(action)
        env_vis.render()
        frame = env_vis.Pixel_Obs()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pixel Observation", frame_bgr)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if hasattr(env_vis, "close"):
        env_vis.close()

def save_stats(stats_data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Best', 'Average', 'Min'])
        for gen, (best, avg, min_val) in enumerate(zip(*stats_data)):
            writer.writerow([gen+1, best, avg, min_val])
    print(f"Statistics saved to {filename}")

def save_checkpoint(population, best_individual, gen, stats_data=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "gp_checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    
    pop_file = os.path.join(save_dir, f"population_gen{gen}_{timestamp}.pkl")
    with open(pop_file, 'wb') as f:
        pickle.dump(population, f)
    
    best_file = os.path.join(save_dir, f"best_individual_gen{gen}_{timestamp}.pkl")
    with open(best_file, 'wb') as f:
        pickle.dump(best_individual, f)
    
    if stats_data:
        stats_file = os.path.join(save_dir, f"stats_gen{gen}_{timestamp}.csv")
        save_stats(stats_data, stats_file)
    
    print(f"Checkpoint saved : {gen}")
    return pop_file, best_file

def load_checkpoint(population_file, best_file=None):
    with open(population_file, 'rb') as f:
        population = pickle.load(f)
    
    best_individual = None
    if best_file:
        with open(best_file, 'rb') as f:
            best_individual = pickle.load(f)
    
    print(f"Loaded checkpoint from {population_file}")
    return population, best_individual

def main():
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=50)
    NGEN = 2000
    CXPB, MUTPB = 0.5, 0.25
    
    # If you want to resume from a checkpoint, uncomment these lines
    # pop, _ = load_checkpoint("gp_checkpoints/population_gen25_20250301_120000.pkl")

    best_scores = []
    avg_scores = []
    min_scores = []

    for gen in tqdm(range(NGEN), desc="Générations"):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        best_ind = tools.selBest(pop, 1)[0]
        current_fits = [ind.fitness.values[0] for ind in pop]

        best_scores.append(best_ind.fitness.values[0])
        avg_scores.append(np.mean(current_fits))
        min_scores.append(np.min(current_fits))

        print(f"\nGénération {gen+1}, "
              f"Score max = {best_scores[-1]:.2f}, "
              f"Score moyen = {avg_scores[-1]:.2f}, "
              f"Score min = {min_scores[-1]:.2f}, "
              f"Taille moyen = {np.mean([len(ind) for ind in pop]):.2f}")
        
        if (gen + 1) % 10 == 0 or gen == NGEN - 1:
            save_checkpoint(pop, best_ind, gen + 1, 
                           stats_data=(best_scores, avg_scores, min_scores))
            # display_game(best_ind)

        if best_scores[-1] > 860:
            save_checkpoint(pop, best_ind, gen + 1, 
                           stats_data=(best_scores, avg_scores, min_scores))
            break

    pool.close()
    pool.join()

    plt.figure()
    plt.plot(range(1, NGEN+1), best_scores, label='Max')
    plt.plot(range(1, NGEN+1), avg_scores, label='Moyen')
    plt.plot(range(1, NGEN+1), min_scores, label='Min')
    plt.xlabel("Génération")
    plt.ylabel("Score")
    plt.title("Évolution des scores")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "gp_results"
    os.makedirs(save_dir, exist_ok=True)
    
    stats_file = os.path.join(save_dir, f"final_stats_{timestamp}.csv")
    save_stats((best_scores, avg_scores, min_scores), stats_file)
    
    best_file = os.path.join(save_dir, f"best_individual_{timestamp}.pkl")
    with open(best_file, 'wb') as f:
        pickle.dump(best_ind, f)
    
    expr_file = os.path.join(save_dir, f"best_expression_{timestamp}.txt")
    with open(expr_file, 'w') as f:
        f.write(str(best_ind))
    
    print(f"Final results : {save_dir}")
    
    return best_ind

if __name__ == "__main__":
    start_time = time.time()
    best_ind = main()
    print(f"Evolution Time: {time.time() - start_time:.2f}s")
    print("Best :", best_ind)
    cv2.destroyAllWindows()
    plt.close()
