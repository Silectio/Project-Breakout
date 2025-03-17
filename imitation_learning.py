import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from env import EnvBreakout, concatenate_human_dataset
from agent_big import DDQNAgent 
from tqdm import tqdm
import optuna

def pretrain_agent_with_demonstrations(agent, demo_data, num_epochs=10, batch_size=32):
    """
    Pretrains the DDQN agent using demonstration data collected from human input.
    
    The demonstration data is assumed to be a list of episodes, where each episode
    is a list of transitions: (state, action, reward, next_state, done).
    
    Parameters:
      agent: The DDQNAgent instance.
      dataset_file: Path to the saved demonstration dataset (pickle file).
      num_epochs: Number of epochs to iterate over the dataset.
      batch_size: Batch size used for each gradient update.
    """

    # Flatten the dataset: from list of episodes to list of transitions
    transitions = [transition for episode in demo_data for transition in episode]
    print(f"Loaded {len(transitions)} transitions from demonstrations.")

    for epoch in range(num_epochs):
        random.shuffle(transitions)
        losses = []
        # Process the transitions in mini-batches
        for i in range(0, len(transitions), batch_size):
            batch = transitions[i:i+batch_size]
            # Unpack batch transitions: each is (state, action, reward, next_state, done)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to PyTorch tensors
            states_t      = torch.FloatTensor(np.array(states)).to(agent.device)
            actions_t     = torch.LongTensor(actions).to(agent.device)
            rewards_t     = torch.FloatTensor(rewards).to(agent.device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(agent.device)
            dones_t       = torch.FloatTensor(dones).to(agent.device)
            
            # Compute current Q-values for the taken actions.
            q_values = agent.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values using the target network.
            with torch.no_grad():
                next_q_values = agent.target_model(next_states_t).max(1)[0]
                target = rewards_t + (1 - dones_t) * agent.gamma * next_q_values
            
            loss = F.smooth_l1_loss(q_values, target)
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)

def objective(trial : optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    max_memory = 100000
    target_update_interval = trial.suggest_int("target_update_interval", 5, 50)
    num_epochs = trial.suggest_int("num_epochs", 10, 50)
    batch_size = trial.suggest_int("batch_size", 32, 512)
    
    env = EnvBreakout(render_mode=None, clipping=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        batch_size=1024,
        max_memory=max_memory,
        target_update_interval=target_update_interval,
        epsilon=0
    )
    
    demo_data = concatenate_human_dataset('human_datasets/')
    pretrain_agent_with_demonstrations(
        agent, demo_data=demo_data, 
        num_epochs=num_epochs, 
        batch_size=batch_size
    )
    
    # Evaluate the agent
    avg_step_count = []
    for episode in range(50):
        done = False
        truncated = False
        agent.model.eval()
        agent.target_model.eval()
        state, info = env.reset()
        num_step = 0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            num_step += 1

        avg_step_count.append(num_step)
    env.close()

    # The metric we want to maximize
    score = np.mean([i for i in avg_step_count])
    agent.save(f"trial_{trial._trial_id}")

    return score


    
if __name__ == '__main__':
    #This params has been determined by optuna tuning on the highest average step count, 
    #this to shorten as much as possible the step where the agent is fully naive on further RL training
    """
    # optuna-dashboard sqlite:///imitation_learning_study.db
    study = optuna.create_study(direction="maximize", storage = "sqlite:///imitation_learning_study.db")
    study.optimize(objective, n_trials=50, n_jobs=8)
    print("Best hyperparams:", study.best_params)
    print("Best value:", study.best_value)
    print("Best trial:", study.best_trial)

    """
    
    params = {'lr': 0.00037528128040352634, 'target_update_interval': 49, 'num_epochs': 29, 'batch_size': 463}
    
    env = EnvBreakout(render_mode=None, clipping=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the DDQN agent
    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=params['lr'],
        batch_size=1024,
        max_memory=100000,
        target_update_interval=params['target_update_interval'],
        epsilon=0
    )
    

    demo_data = concatenate_human_dataset('human_datasets/')
    print("Training on : ", len(demo_data), "episodes")
    # Pretrain the DDQN agent on the demonstration dataset
    pretrain_agent_with_demonstrations(agent, demo_data=demo_data, num_epochs=params['num_epochs'], batch_size = params['batch_size'])
    agent.save('checkpoint_pretrained_imitation')
    
    # At this point, the agent has been initialized with human demonstrations.
    
    agent.load("./checkpoint_pretrained_imitation")
    avg_reward = []
    avg_step_count = []
    for episode in tqdm(range(100), total=100):
    #let's eval the imitation policy 
        done = False
        truncated = False
        agent.model.eval()
        agent.target_model.eval()
        state, info = env.reset()
        num_step = 0
        total_reward = 0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _, real_reward = env.step(action, return_real_reward=True)
            state = next_state
            total_reward+=reward
            num_step+=1

        avg_reward.append(total_reward)
        avg_step_count.append(num_step)
        
    print(" Avg Reward : ", np.mean(avg_reward), " Avg step : ", np.mean([ i for i in avg_step_count]))
    