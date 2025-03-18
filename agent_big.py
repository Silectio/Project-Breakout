import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from modular_mlp import ModularMLP
import time
import torch.nn.functional as F
from env import EnvBreakout
import random
from PrioritizedReplayMemory import PrioritizedReplayMemory
from datetime import timedelta
from tqdm import tqdm
from imitation_learning import concatenate_human_dataset
import optuna

def compute_epsilon_decay(eps_start, eps_end, num_episodes):

    epsilon_decay = (eps_end / eps_start) ** (1 / num_episodes)
    return epsilon_decay

class DDQNAgent:
    def __init__(
        self, state_size, action_size, gamma=0.99, lr=3e-4,
        batch_size=128, max_memory=500, epsilon=1.0,
        epsilon_min=0.01, epsilon_decay=0.99965, max_grad_norm=1.0,
        target_update_interval=2000, demo_ratio = 0
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self.learn_steps = 0
        self.demo_ratio = demo_ratio
        self.loss = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network
        self.model = ModularMLP(input_dim=state_size, output_dim=action_size, num_hidden_layers=3, hidden_dim=64).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)

        # Target network
        self.target_model = ModularMLP(input_dim=state_size, output_dim=action_size, num_hidden_layers=3, hidden_dim=64).to(self.device)
        self.update_target_network()  # copy weights initially
        self.target_model.eval()

        #self.loss_fn = nn.MSELoss(reduction='none')  # or use SmoothL1Loss(reduction='none')
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.memory = PrioritizedReplayMemory(max_memory)
        #self.memory = SumTreeReplayMemory(capacity=max_memory, alpha=0.6, beta=0.4, beta_increment=3e-4)
        
        self.demo_memory = None


    def act(self, state, eval_mode=False):
        # Epsilon-greedy strategy
        if (not eval_mode) and (random.random() < self.epsilon):
            return random.randint(0, self.action_size - 1)

        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # shape [1, state_dim]
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values, dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))
    
    
    def load_demo_data(self, demo_transitions):
        """
        Charge les données de démonstration dans une mémoire dédiée, 
        dont la taille est égale au nombre de transitions de démonstration.
        """
        if self.demo_memory is None:
            capacity_demo = len(demo_transitions)
            """
            self.demo_memory = SumTreeReplayMemory(
                capacity=capacity_demo,
                alpha=0.6,
                beta=0.4,
                beta_increment=3e-4
            )
            """
            self.demo_memory = PrioritizedReplayMemory(capacity_demo)
        for transition in demo_transitions:
            # On peut mettre une priorité fixe (ex: 5.0) ou variable
            self.demo_memory.store(transition)

    def train_step(self):
        """
        Exemple de train_step qui échantillonne un mini-batch mêlant
        un sous-ensemble de transitions issues de la démonstration
        et un sous-ensemble des transitions classiques de replay.
        """
        # Empêche de démarrer l'entraînement si peu de données
        if len(self.memory) < self.batch_size :
            return

        
        # Si vous avez déjà chargé la demo_memory :
        if self.demo_memory is not None and len(self.demo_memory) > 0:
            batch_size_demo = int(self.batch_size * self.demo_ratio)
        else:
            batch_size_demo = 0

        batch_size_main = self.batch_size - batch_size_demo

        # Échantillonner dans la mémoire principale
        batch_main, indices_main, weights_main = self.memory.sample(batch_size_main)

        if batch_size_demo > 0:
            # Échantillonner dans la mémoire de démonstration
            batch_demo, indices_demo, weights_demo = self.demo_memory.sample(batch_size_demo)

            # Fusionner les deux sous-batchs
            batch = batch_main + batch_demo
            #indices = np.concatenate((indices_main, indices_demo), axis=0)
            weights = np.concatenate((weights_main, weights_demo), axis=0)
        else:
            batch = batch_main
            #indices = indices_main
            weights = weights_main
        
        # Conversion en tenseurs
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t      = torch.stack(states).to(self.device)
        actions_t     = torch.stack(actions).to(self.device)
        rewards_t     = torch.stack(rewards).clip(-1,0.1).to(self.device)
        next_states_t = torch.stack(next_states).to(self.device)
        dones_t       = torch.stack(dones).to(self.device)

        weights_t     = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Calcul Q-values online
        all_q = self.model(states_t)
        q_values = all_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)
            q_next = self.target_model(next_states_t).gather(1, next_actions).squeeze(1)
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        td_errors = q_target - q_values
        elementwise_loss = self.loss_fn(q_values, q_target)
        loss = (elementwise_loss * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        # Mise à jour des priorités
        new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-5

        if batch_size_main > 0 :
            # MàJ des priorités de la mémoire principale
            self.memory.update_priorities(indices_main, new_priorities[:batch_size_main])
        
        # S'il y a une partie démonstration, mettre à jour aussi leurs priorités
        if batch_size_demo > 0:
            self.demo_memory.update_priorities(indices_demo, new_priorities[batch_size_main:])

        self.loss = loss.item()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.update_target_network()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        with torch.no_grad():
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder):
        torch.save(self.model.state_dict(), os.path.join(folder, "model.pth"))
        torch.save(self.target_model.state_dict(), os.path.join(folder, "target_model.pth"))

    def load(self, folder):
        self.model.load_state_dict(torch.load(os.path.join(folder, "model.pth")))
        self.target_model.load_state_dict(torch.load(os.path.join(folder, "target_model.pth")))



import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta


def print_agent_vs_env(agent : DDQNAgent):
    env = EnvBreakout(render_mode='human')
    step = 0
    done = False
    truncated = False
    state, _ = env.reset()
    while step < 100000 or not done:
        action = agent.act(state, eval_mode=True)
        state, reward, done, truncated, info = env.step(action)
        
        env.render()
        time.sleep(0.01)
        
        step +=1
        
        fire_result = env.step_fire_if_needed(return_real_reward=True)
        if fire_result is not None:
            state, reward, done, truncated, info, real_reward = fire_result
        if done or truncated :
            break
        
    env.close()
    return

def train_no_parallel(agent : DDQNAgent, env : EnvBreakout, total_episodes=10000, random_episodes=1000):
    print("cuda" if torch.cuda.is_available() else "cpu")

    agent.model.train()
    start_time = time.time()
    
    # Compteur d'épisodes complétés
    completed_episodes = 0

    # Listes pour suivre l'évolution des métriques à chaque épisode
    loss_list = []
    reward_list = []
    step_count_list = []

    # Listes de moyennes pour la traçage (moyennes glissantes)
    avg_loss_list = []
    avg_reward_list = []
    avg_step_count_list = []

    # Gestion des "victoires" (ou toute autre condition de succès)
    win_count = 0
    win_count_list = []

    # Barre de progression
    pbar = tqdm(total=total_episodes, leave=True)

    while completed_episodes < total_episodes:
        # Réinitialisation de l'environnement et des variables de suivi
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_real_reward = 0.0
        step_count = 0
        avg_loss = 0.0

        # Boucle d’interactions agent-environnement pour un seul épisode
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _, real_reward = env.step(action, return_real_reward=True)


            # Récupération de la perte si définie
            if agent.loss is not None:
                avg_loss += agent.loss

            # Stockage de l'expérience et entraînement de l'agent
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            # Mise à jour des compteurs
            state = next_state
            episode_reward += reward
            episode_real_reward += real_reward
            step_count += 1

            # Limite arbitraire sur le nombre de pas par épisode
            if step_count >= 10000:
                break
            fire_result = env.step_fire_if_needed(return_real_reward=True)
            if fire_result is not None:
                next_state, reward, done, truncated, info, real_reward = fire_result
                
        if agent.demo_ratio >0:
            agent.demo_ratio = max(0.1, agent.demo_ratio * compute_epsilon_decay(0.5,0.05 , 500))
        # Fin de l'épisode
        completed_episodes += 1
        pbar.update(1)

        # Mise à jour d'epsilon après un certain nombre d'épisodes "random"
        if completed_episodes >= random_episodes:
            agent.update_epsilon()

        # Calcul des moyennes et ajouts dans les listes
        # (on divise la perte cumulée par le nombre de pas pour avoir une moyenne par étape)
        mean_loss_per_step = avg_loss / step_count if step_count > 0 else 0.0
        loss_list.append(mean_loss_per_step)
        reward_list.append(episode_reward)
        step_count_list.append(step_count)

        # Gestion des victoires (exemple : real_reward > 850)
        if episode_real_reward > 850:
            win_count += 1

        # Impression et calcul de moyennes glissantes tous les print_interval épisodes
        if (completed_episodes > 0) and completed_episodes % 10 ==0:
            elapsed_time = time.time() - start_time
            print(f"\nFin épisode {completed_episodes}")
            print(f"Récompense de l'épisode : {episode_reward:.2f}")
            print(f"Récompense réelle de l'épisode : {episode_real_reward:.2f}")
            print(f"Perte (moyenne par step, dernier épisode) : {mean_loss_per_step:.6f}")
            print(f"Nombre d'étapes dans l'épisode : {step_count}")
            print(f"Epsilon : {agent.epsilon:.3f}")
            print(f"Temps écoulé : {str(timedelta(seconds=elapsed_time))}")

            # Calcul de la moyenne sur les 100 derniers épisodes ou moins
            window = min(10, len(loss_list))
            avg_loss_100 = np.mean(loss_list[-window:]) if window > 0 else 0.0
            avg_reward_100 = np.mean(reward_list[-window:]) if window > 0 else 0.0
            avg_steps_100 = np.mean(step_count_list[-window:]) if window > 0 else 0.0

            avg_loss_list.append(avg_loss_100)
            avg_reward_list.append(avg_reward_100)
            avg_step_count_list.append(avg_steps_100)
            win_count_list.append(win_count)
            
            print(f"Demo ratio : {agent.demo_ratio:.3f}")
            print(f'LR : {agent.scheduler.get_last_lr()}')
            print(f"Moyenne des pertes sur les {window} derniers épisodes : {avg_loss_100:.6f}")
            print(f"Moyenne des récompenses sur les {window} derniers épisodes : {avg_reward_100:.2f}")
            print(f"Moyenne des étapes sur les {window} derniers épisodes : {avg_steps_100:.2f}")
            print(f"Victoires cumulées : {win_count}")
            print("=" * 70)

        # Sauvegarde des checkpoints et tracés tous les save_interval épisodes
        if (completed_episodes > 0)and completed_episodes % 10 ==0:
            # Création du dossier de sauvegarde si besoin
            save_folder = "checkpoint_DDQN_single"
            os.makedirs(save_folder, exist_ok=True)
            agent.save(save_folder)

            # Préparation du tracé
            # Pour rester cohérent avec train_parallel, on trace les moyennes glissantes (et pas chaque épisode)
            # On crée l'axe des X en fonction du nombre de points dans avg_loss_list
            x_list = np.linspace(0, completed_episodes, len(avg_loss_list))

            fig = plt.figure(figsize=(24, 12))
            fig.suptitle(f"Évolution jusqu'à l'épisode {completed_episodes}")
            axes = fig.subplots(2, 2)

            # Pertes (moyenne glissante) - échelle log
            axes[0, 0].plot(x_list, avg_loss_list)
            axes[0, 0].set(xscale="linear", yscale="log", xlabel="Épisode", ylabel="Perte (moyenne)",
                           title="Pertes (moyenne glissante, échelle log)")

            # Récompenses (moyenne glissante)
            axes[1, 0].plot(x_list, avg_reward_list)
            axes[1, 0].set(xlabel="Épisode", ylabel="Récompense (moyenne)",
                           title="Récompenses (moyenne glissante)")

            # Nombre d'étapes (moyenne glissante)
            axes[0, 1].plot(x_list, avg_step_count_list)
            axes[0, 1].set(xlabel="Épisode", ylabel="Nombre d'étapes (moy.)",
                           title="Nombre d'étapes (moyenne glissante)")

            # Nombre de victoires cumulées
            axes[1, 1].plot(x_list, win_count_list)
            axes[1, 1].set(xlabel="Épisode", ylabel="Victoires cumulées",
                           title="Victoires cumulées")

            plt.tight_layout()
            fig.savefig('training_plot_single.png')
            plt.close()
        
        if completed_episodes%100 ==  0 : 
            print_agent_vs_env(agent=agent)

    pbar.close()
    print("\nEntraînement terminé !")
    print(f"{total_episodes} épisodes complétés.")

    # On peut retourner la liste des récompenses ou tout autre tableau de stats
    return reward_list

def train_no_parallel_mute(agent : DDQNAgent, env : EnvBreakout, total_episodes=10000, random_episodes=1000, trial : optuna.trial.Trial =None):
    print("cuda" if torch.cuda.is_available() else "cpu")

    agent.model.train()
    
    # Compteur d'épisodes complétés
    completed_episodes = 0

    # Listes pour suivre l'évolution des métriques à chaque épisode
    loss_list = []
    reward_list = []
    step_count_list = []

    # Listes de moyennes pour la traçage (moyennes glissantes)
    avg_loss_list = []
    avg_reward_list = []
    avg_step_count_list = []

    # Gestion des "victoires" (ou toute autre condition de succès)
    win_count = 0
    win_count_list = []

    while completed_episodes < total_episodes:
        # Réinitialisation de l'environnement et des variables de suivi
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_real_reward = 0.0
        step_count = 0
        avg_loss = 0.0

        # Boucle d’interactions agent-environnement pour un seul épisode
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _, real_reward = env.step(action, return_real_reward=True)

            # Récupération de la perte si définie
            if agent.loss is not None:
                avg_loss += agent.loss

            # Stockage de l'expérience et entraînement de l'agent
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            # Mise à jour des compteurs
            state = next_state
            episode_reward += reward
            episode_real_reward += real_reward
            step_count += 1

            # Limite arbitraire sur le nombre de pas par épisode
            if step_count >= 10000:
                break
                
        agent.demo_ratio = max(0.1, agent.demo_ratio * compute_epsilon_decay(1,0.1 , total_episodes))
        # Fin de l'épisode
        completed_episodes += 1

        # Mise à jour d'epsilon après un certain nombre d'épisodes "random"
        if completed_episodes >= random_episodes:
            agent.update_epsilon()

        # Calcul des moyennes et ajouts dans les listes
        # (on divise la perte cumulée par le nombre de pas pour avoir une moyenne par étape)
        mean_loss_per_step = avg_loss / step_count if step_count > 0 else 0.0
        loss_list.append(mean_loss_per_step)
        reward_list.append(episode_reward)
        step_count_list.append(step_count)

        # Gestion des victoires (exemple : real_reward > 850)
        if episode_real_reward > 850:
            win_count += 1

        # Impression et calcul de moyennes glissantes tous les print_interval épisodes
        if (completed_episodes > 0):

            # Calcul de la moyenne sur les 10 derniers épisodes ou moins
            window = min(10, len(loss_list))
            avg_loss_100 = np.mean(loss_list[-window:]) if window > 0 else 0.0
            avg_reward_100 = np.mean(reward_list[-window:]) if window > 0 else 0.0
            avg_steps_100 = np.mean(step_count_list[-window:]) if window > 0 else 0.0

            avg_loss_list.append(avg_loss_100)
            avg_reward_list.append(avg_reward_100)
            avg_step_count_list.append(avg_steps_100)
            win_count_list.append(win_count)
            if trial is not None and completed_episodes % 10 == 0:
                # Report average reward over the last 100 episodes (or current if fewer)
                recent_rewards = reward_list[-10:] if len(reward_list) >= 10 else reward_list
                intermediate_value = np.mean(recent_rewards)
                trial.report(intermediate_value, completed_episodes)

    print("\nEntraînement terminé !")
    print(f"{total_episodes} épisodes complétés.")

    # On peut retourner la liste des récompenses ou tout autre tableau de stats
    return reward_list


def train_parallel(agent, envs, total_episodes = 10000, random_episodes = 1000, num_envs=4, print_step = 10):
    print("cuda" if torch.cuda.is_available() else "cpu")

    # Nombre d'environnements parallèles

    agent.model.train()
    completed_episodes = 0

    # Listes de statistiques par épisode pour le traçage
    avg_loss_list = []
    avg_reward_list = []
    avg_step_count_list = []
    loss_list = []
    reward_list = []
    step_count_list = []
    win_count = 0
    win_count_list = []

    # Initialisation des états via np.array
    states_list = []
    for env in envs:
        s, _ = env.reset()
        states_list.append(s)
    states = np.array(states_list)  # Conversion explicite en np.array

    # Initialisation des statistiques par environnement
    ep_rewards       = np.zeros(num_envs, dtype=np.float32)
    ep_real_rewards  = np.zeros(num_envs, dtype=np.float32)
    step_counts      = np.zeros(num_envs, dtype=np.int32)
    done_flags       = np.array([False] * num_envs)
    truncated_flags  = np.array([False] * num_envs)

    print_interval = 10
    save_interval = 10
    start_time = time.time()

    pbar = tqdm(total=total_episodes, leave=True)
    while completed_episodes < total_episodes:
        # Sélection des actions pour chaque environnement actif
        actions = []
        for i in range(num_envs):
            if not done_flags[i] and not truncated_flags[i]:
                actions.append(agent.act(states[i]))
            else:
                #reset the environment
                s, _ = envs[i].reset()
                states[i] = s
                actions.append(agent.act(states[i]))

        # Passage à l'étape suivante dans chaque environnement
        next_states_list = []
        rewards         = np.zeros(num_envs, dtype=np.float32)
        real_rewards    = np.zeros(num_envs, dtype=np.float32)
        new_done_flags      = np.array([False] * num_envs)
        new_truncated_flags = np.array([False] * num_envs)

        for i in range(num_envs):
            if not done_flags[i] and not truncated_flags[i]:
                ns, reward, done, truncated, _, real_reward = envs[i].step(actions[i], return_real_reward=True)
                next_states_list.append(ns)
                rewards[i]      = reward
                real_rewards[i] = real_reward
                new_done_flags[i]      = done
                new_truncated_flags[i] = truncated
            else:
                next_states_list.append(states[i])
                rewards[i] = 0.0
                real_rewards[i] = 0.0
                new_done_flags[i]      = done_flags[i]
                new_truncated_flags[i] = truncated_flags[i]

        # Stockage des expériences pour chaque environnement actif
        for i in range(num_envs):
            if not done_flags[i] and not truncated_flags[i]:
                agent.store_experience(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states_list[i],
                    new_done_flags[i]
                )
        agent.train_step()

        # Mise à jour des statistiques par environnement et détection de fin d'épisode
        for i in range(num_envs):
            if not done_flags[i] and not truncated_flags[i]:
                ep_rewards[i]      += rewards[i]
                ep_real_rewards[i] += real_rewards[i]
                step_counts[i]     += 1

            done_flags[i]      = new_done_flags[i]
            truncated_flags[i] = new_truncated_flags[i]

            # Fin d'épisode : si l'environnement est terminé, tronqué ou dépasse 10000 étapes
            if done_flags[i] or truncated_flags[i] or (step_counts[i] >= 10000):
                completed_episodes += 1
                pbar.update(1)
                agent.demo_ratio = max(0.05, agent.demo_ratio * compute_epsilon_decay(0.5,0.05 , total_episodes))
                if completed_episodes >= random_episodes:
                    agent.update_epsilon()
                if ep_real_rewards[i] > 850:
                    win_count+=1
                episode_reward  = ep_rewards[i]
                episode_real_reward = ep_real_rewards[i]
                episode_steps   = step_counts[i]
                episode_loss    = agent.loss if agent.loss is not None else 0.0

                loss_list.append(episode_loss)
                reward_list.append(episode_reward)
                step_count_list.append(episode_steps)

                if (completed_episodes > 0) and completed_episodes % print_step ==0:
                    elapsed_time = time.time() - start_time
                    print(f"\nFin épisode {completed_episodes}")
                    print(f"Récompense de l'épisode : {episode_reward}")
                    print(f"Récompense réelle de l'épisode : {episode_real_reward}")
                    print(f"Perte (dernière) : {episode_loss:.6f}")
                    print(f"Nombre d'étapes dans l'épisode : {episode_steps}")
                    print(f"Epsilon : {agent.epsilon:.3f}")
                    print(f"Temps écoulé : {str(timedelta(seconds=elapsed_time))}")

                    if completed_episodes >= print_step:
                        avg_loss = np.mean(loss_list[-print_step:])
                        avg_reward = np.mean(reward_list[-print_step:])
                        avg_step_count = np.mean(step_count_list[-print_step:])
                    else:
                        avg_loss = np.mean(loss_list)
                        avg_reward = np.mean(reward_list)
                        avg_step_count = np.mean(step_count_list)
                    avg_loss_list.append(avg_loss)
                    avg_reward_list.append(avg_reward)
                    avg_step_count_list.append(avg_step_count)
                    win_count_list.append(win_count)

                    print(f"Demo ratio : {agent.demo_ratio:.3f}")
                    print(f'LR : {agent.scheduler.get_last_lr()}')
                    print(f"Moyenne des pertes sur les 100 derniers épisodes : {avg_loss:.6f}")
                    print(f"Moyenne des récompenses sur les 100 derniers épisodes : {avg_reward:.6f}")
                    print(f"Moyenne des étapes sur les 100 derniers épisodes : {avg_step_count:.6f}")
                    print(f"Victoires cumulées : {win_count}")
                    print("=" * 70)

                    save_folder = "checkpoint_DDQN_parallel"
                    os.makedirs(save_folder, exist_ok=True)
                    agent.save(save_folder)

                    
                    x_list= np.linspace(0, completed_episodes, len(avg_loss_list))
                    fig = plt.figure(figsize=(24, 12))
                    fig.suptitle("Till episode {completed_episodes}")
                    axes = fig.subplots(2, 2)
                    axes[0,0].plot(x_list, avg_loss_list)
                    #plot average loss for each ten episodes
                    axes[0,0].set(xscale="linear", yscale="log", xlabel="Épisode", ylabel="Perte", title=f"Perte (échelle log)")

                    axes[1,0].plot(x_list, avg_reward_list)
                    axes[1,0].set(xlabel="Épisode", ylabel="Récompense", title=f"Récompense")

                    axes[0,1].plot(x_list, avg_step_count_list)
                    axes[0,1].set(xlabel="Épisode", ylabel="Nombre d'étapes", title=f"Nombre d'étapes")

                    axes[1,1].plot(x_list, win_count_list)
                    axes[1,1].set(xlabel="Épisode", ylabel="Victoires cumulées", title=f"Victoires cumulées")
                    plt.tight_layout()
                    fig.savefig('training_plot_parallel.png')
                    plt.close()

                # Réinitialisation des compteurs pour cet environnement
                ep_rewards[i]      = 0.0
                ep_real_rewards[i] = 0.0
                step_counts[i]     = 0
                done_flags[i]      = False
                truncated_flags[i] = False

                s, _ = envs[i].reset()
                next_states_list[i] = s

        # Mise à jour des états en utilisant np.array
        states = np.array(next_states_list)
    pbar.close()
    print("\nEntraînement terminé !")
    print(f"{total_episodes} épisodes complétés.")
    return reward_list

def train_parallel_mute(agent, envs, total_episodes=10000, random_episodes=1000, num_envs=4, trial : optuna.trial.Trial =None):
    agent.model.train()
    completed_episodes = 0

    loss_list = []
    reward_list = []
    step_count_list = []

    # Initialisation des états
    states = np.array([env.reset()[0] for env in envs])

    # Initialisation des statistiques par environnement
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_real_rewards = np.zeros(num_envs, dtype=np.float32)
    step_counts = np.zeros(num_envs, dtype=np.int32)
    done_flags = np.zeros(num_envs, dtype=bool)
    truncated_flags = np.zeros(num_envs, dtype=bool)

    #pbar = tqdm(total=total_episodes, miniters=200)
    while completed_episodes < total_episodes:
        # Réinitialisation des environnements terminés
        for i in range(num_envs):
            if done_flags[i] or truncated_flags[i]:
                states[i] = envs[i].reset()[0]

        # Calcul des actions pour tous les environnements
        actions = [agent.act(states[i]) for i in range(num_envs)]

        # Passage à l'étape suivante dans chaque environnement
        next_states = np.empty_like(states)
        rewards = np.zeros(num_envs, dtype=np.float32)
        real_rewards = np.zeros(num_envs, dtype=np.float32)
        new_done_flags = np.zeros(num_envs, dtype=bool)
        new_truncated_flags = np.zeros(num_envs, dtype=bool)

        for i in range(num_envs):
            # Si l'environnement est actif, on avance d'un pas
            if not (done_flags[i] or truncated_flags[i]):
                ns, reward, done, truncated, _, real_reward = envs[i].step(actions[i], return_real_reward=True)
                next_states[i] = ns
                rewards[i] = reward
                real_rewards[i] = real_reward
                new_done_flags[i] = done
                new_truncated_flags[i] = truncated
            else:
                next_states[i] = states[i]
        
        # Stockage des expériences pour les environnements actifs
        for i in range(num_envs):
            if not (done_flags[i] or truncated_flags[i]):
                agent.store_experience(states[i], actions[i], rewards[i], next_states[i], new_done_flags[i])

        agent.train_step()
        
        # Mise à jour des statistiques et détection de fin d'épisode
        for i in range(num_envs):
            if not (done_flags[i] or truncated_flags[i]):
                ep_rewards[i] += rewards[i]
                ep_real_rewards[i] += real_rewards[i]
                step_counts[i] += 1

            # Mise à jour des flags
            done_flags[i] = new_done_flags[i]
            truncated_flags[i] = new_truncated_flags[i]

            # Fin d'épisode
            if done_flags[i] or truncated_flags[i] or step_counts[i] >= 10000:
                completed_episodes += 1
                agent.demo_ratio = max(0.1, agent.demo_ratio * compute_epsilon_decay(1,0.1 , 1000))
                #pbar.update(1)
                if completed_episodes >= random_episodes:
                    agent.update_epsilon()

                # Stockage des statistiques de l'épisode
                loss_list.append(agent.loss if agent.loss is not None else 0.0)
                reward_list.append(ep_rewards[i])
                step_count_list.append(step_counts[i])

                if trial is not None and completed_episodes % 10 == 0:
                    # Report average reward over the last 100 episodes (or current if fewer)
                    recent_rewards = reward_list[-10:] if len(reward_list) >= 10 else reward_list
                    intermediate_value = np.mean(recent_rewards)
                    trial.report(intermediate_value, completed_episodes)

                # Réinitialisation de l'environnement et des compteurs
                ep_rewards[i] = 0.0
                ep_real_rewards[i] = 0.0
                step_counts[i] = 0
                done_flags[i] = False
                truncated_flags[i] = False
                states[i] = envs[i].reset()[0]
            else:
                states[i] = next_states[i]
    #pbar.close()
    return reward_list

def objective(trial: optuna.trial.Trial):
    lr=trial.suggest_loguniform('lr', 1e-5, 3e-4)
    target_update_interval=trial.suggest_int('target_update_interval', 10, 100, step=10)
    start_epsilon = trial.suggest_float('start_epsilon', 0.2, 0.6, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])

    num_envs= 8
    total_episodes = 10000
    random_episodes = 500
    #envs = [EnvBreakout(render_mode=None, clipping=True) for _ in range(num_envs)]
    envs = EnvBreakout(render_mode=None, clipping=True)
    # Création de l'agent en se basant sur le premier environnement
    agent = DDQNAgent(
        state_size=envs.observation_space.shape[0],
        action_size=envs.action_space.n,
        lr=lr,
        batch_size=batch_size,
        max_memory=500000,
        target_update_interval=target_update_interval,
        epsilon_decay=compute_epsilon_decay(start_epsilon, 0.01, total_episodes-random_episodes-100),
        epsilon=start_epsilon,
        max_grad_norm=3
    )
    agent.gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)
    #using pretrained weights to fasten training
    #agent.load('checkpoint_pretrained_imitation')

    #preload memory with demonstration data
    demo_data = concatenate_human_dataset('human_datasets/')
    transitions = [transition for episode in demo_data for transition in episode]

    # Apply agent.RAM_Obs to each observation in transitions
    transitions = [(EnvBreakout.RAM_Obs(transition[0]), transition[1], transition[2], EnvBreakout.RAM_Obs(transition[3]), transition[4]) for transition in transitions]

    agent.load_demo_data(transitions) 

    try :
        train_parallel_mute(agent=  agent, envs= envs, total_episodes=total_episodes, random_episodes=random_episodes, num_envs=num_envs, trial= trial)
        #train_no_parallel_mute(agent=agent, env=envs, total_episodes=total_episodes, random_episodes=random_episodes, trial=trial)
    except Exception as e:
        print(f"Trial : {trial.number}, failed with error : \n\n {e}")

    agent.model.eval()
    agent.target_model.eval()
    reward_list =[]
    step_list = []
    for k in range(100):
        env = EnvBreakout(render_mode=None, clipping=True)
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        while not done or not truncated:
            step +=1
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        step_list.append(step)
        reward_list.append(total_reward)

    return np.mean(reward_list)

if __name__ == "__main__":
    """
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="optuna_study",
        storage="sqlite:///optuna_study.db",  # Using SQLite for persistence
        load_if_exists=True,
        direction="maximize"
        )

    # optuna-dashboard sqlite:///optuna_study.db
    study.optimize(objective, n_trials=50, n_jobs = 8, show_progress_bar=True)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    env = EnvBreakout(render_mode=None, clipping=True)
    agent = DDQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=3e-4,
        batch_size=1024,
        max_memory=500000,
        target_update_interval=20,
        epsilon_decay=compute_epsilon_decay(0.3, 0.01, total_episodes-random_episodes-5000),
        epsilon=0.3
    )
    demo_data = concatenate_human_dataset('human_datasets/')
    transitions = [transition for episode in demo_data for transition in episode]
    for transition in transitions:
        agent.memory.store(transition, priority=5)
    train_no_parallel(agent, env, total_episodes=total_episodes, random_episodes=random_episodes, print_interval=10, save_interval=10)
    """
    
    total_episodes = 2000
    random_episodes = 100
    num_envs= 64
    epsilon=1
    envs = [EnvBreakout(render_mode=None, clipping=True) for _ in range(num_envs)]
    # Création de l'agent en se basant sur le premier environnement
    agent = DDQNAgent(
        state_size=envs[0].observation_space.shape[0],
        action_size=envs[0].action_space.n,
        lr=1e-3,
        batch_size=1024,
        max_memory=100000,
        target_update_interval=10000,
        epsilon=epsilon,
        epsilon_decay=compute_epsilon_decay(epsilon, 0.01, total_episodes-random_episodes-10),
        max_grad_norm=1,
        demo_ratio=0.8
    )

    #using pretrained weights to fasten training
    #agent.load('checkpoint_pretrained_imitation')

    #preload memory with demonstration data
    demo_data = concatenate_human_dataset('human_datasets/')
    transitions = [transition for episode in demo_data for transition in episode]

    # Apply agent.RAM_Obs to each observation in transitions
    transitions = [(EnvBreakout.RAM_Obs(transition[0]), transition[1], transition[2], EnvBreakout.RAM_Obs(transition[3]), transition[4]) for transition in transitions]

    #agent.load_demo_data(transitions) 

    #train_parallel(agent=  agent, envs= envs, total_episodes=total_episodes, random_episodes=random_episodes, num_envs=num_envs, print_step=100)
    train_no_parallel(agent, EnvBreakout(render_mode=None, clipping=True), total_episodes=total_episodes, random_episodes=random_episodes)

'''
if __name__ == '__main__':
    agent = DDQNAgent(
        state_size=EnvBreakout().observation_space.shape[0],
        action_size=EnvBreakout().action_space.n, epsilon=0)
    agent.load("checkpoint_DDQN_single")
    print_agent_vs_env(agent=agent)
'''
    
    