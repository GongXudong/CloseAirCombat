import sys
import os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from envs.JSBSim.envs import SingleControlEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

my_log_dir = 'exp_logs'


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False


def _t2n(x):
    return x.detach().cpu().numpy()


num_agents = 1
render = True
ego_policy_index = 'latest'
episode_rewards = 0
ego_run_dir = "/home/ucav/PythonProjects/CloseAirCombat/scripts/results/SingleControl/1/heading/ppo/v4/run5"
experiment_name = ego_run_dir.split('/')[-2]

env = SingleControlEnv("1/heading")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))

dt_str = datetime.now().strftime("%Y-%m-%d--%H-%M")

print("Start render")
obs = env.reset()
if render:
    env.render(mode='txt', filepath=f'{my_log_dir}/{experiment_name}_{dt_str}.txt.acmi')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents, 1))
ego_obs = obs

step_list, reward_list, delta_heading_list, delta_alt_list, delta_v_list, action_list = [], [], [], [], [], []

while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)

    obs, rewards, dones, infos = env.step(ego_actions)
    episode_rewards += rewards
    if render:
        env.render(mode='txt', filepath=f'{my_log_dir}/{experiment_name}_{dt_str}.txt.acmi')
    if dones.all():
        print("infos: ", infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]

    print(f"step:{env.current_step}, bloods:{bloods}, rewards: {rewards}, delta heading(deg): {np.degrees(obs[0][0])}, delta alt(m): {obs[0][1]*1000}, delta v(mps): {obs[0][2]*340}")
    print(f"step:{env.current_step}, action: {ego_actions}")

    step_list.append(env.current_step)
    reward_list.append(rewards[0][0])
    delta_heading_list.append(np.degrees(obs[0][0]))
    delta_alt_list.append(obs[0][1]*1000)
    delta_v_list.append(obs[0][2]*340)
    action_list.append(ego_actions[0])
    # enm_obs = obs[num_agents // 2:, ...]
    # ego_obs = obs[:num_agents // 2, ...]
    ego_obs = obs

column_names = ['step', 'reward', 'delta heading(deg)', 'delta alt(m)', 'delta v(mps)', 'actions']
print(len(step_list), len(reward_list), len(delta_heading_list), len(delta_alt_list), len(delta_v_list), len(action_list))
logs_df = pd.DataFrame(data=np.array([step_list, reward_list, delta_heading_list, delta_alt_list, delta_v_list, action_list]).T, columns=column_names)
logs_df.to_csv(f"{my_log_dir}/{experiment_name}_{dt_str}.csv", sep=',', index=False)

print(episode_rewards)

