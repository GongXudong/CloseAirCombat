import numpy as np
import torch
from typing import Callable
from functools import partial

from ..ppo.ppo_policy import PPOPolicy
from .utils import check

norm2 = partial(torch.norm, p=1, dim=1)


class LossCalculator(object):
    def __init__(self, dist_calculator: Callable[[torch.Tensor], torch.Tensor] = norm2):
        """_summary_

        Args:
            dist_calculator (Callable[[torch.Tensor], torch.Tensor], optional): 范数计算器，传入np.array，返回np.float64类型的范数值. Defaults to norm2.
        """
        self.dist_calculator = dist_calculator

    def calc_loss(self, policy: PPOPolicy,
                 action_dist_probs: np.ndarray,
                 obs: np.ndarray,
                 rnn_states_actor: np.ndarray,
                 masks: np.ndarray) -> torch.Tensor:
        """计算根据(obs, rnn_states_actor, masks)计算出的动作的分布与原来的动作分布之间的loss。
        TODO：根据action_dist计算？？？？
        TODO: 根据谁更新谁？？？让obs处的动作，像其边上的？还是让obs附近状态的动作，像obs的？？？？？
        现在的逻辑是让obs处的动作像其边上的。

        Args:
            policy (PPOPolicy): _description_
            action_log_probs (np.ndarray): _description_
            obs (np.ndarray): _description_
            rnn_states_actor (np.ndarray): _description_
            masks (np.ndarray): _description_

        Returns:
            torch.Tensor: _description_
        """
        policy.prep_rollout()
        # n_values, n_actions, n_action_log_probs, n_rnn_states_actor, n_rnn_states_critic = policy.get_actions(obs, rnn_states_actor, masks, deterministic=True)
        new_state_action_dist_probs = policy.get_action_dist_probs(obs, rnn_states_actor, masks)

        policy.prep_training()
        loss = self.dist_calculator(action_dist_probs - new_state_action_dist_probs)
        return loss.mean()


class ActionTemporalSmoothLossCalculator(LossCalculator):

    def __init__(self, dist_calculator: Callable[[torch.Tensor], torch.Tensor] = norm2):
        super().__init__(dist_calculator)

    def __call__(self, policy: PPOPolicy,
                 action_dist_probs: np.ndarray,
                 next_obs: np.ndarray,
                 rnn_next_states_actor: np.ndarray,
                 next_masks: np.ndarray) -> torch.Tensor:

        return self.calc_loss(
            policy=policy, 
            action_dist_probs=action_dist_probs, 
            obs=next_obs, 
            rnn_states_actor=rnn_next_states_actor, 
            masks=next_masks
        )
        

class ActionSpatialSmoothLossCalculator(LossCalculator):

    def __init__(self, 
                 dist_calculator: Callable[[torch.Tensor], torch.Tensor] = norm2, 
                 local_for_sample_new_obs: float = 0.01, 
                 repeat: int = 1):
        
        super().__init__(dist_calculator)
        self.local_for_sample_new_obs = local_for_sample_new_obs  # 采样新状态使用的方差
        self.repeat = repeat  # 采样重复的次数

    def __call__(self, policy: PPOPolicy,
                 action_dist_probs: np.ndarray,
                 obs: np.ndarray, 
                 rnn_states_actor: np.ndarray, 
                 masks: np.ndarray) -> torch.Tensor:
        
        loss_list = []
        for i in range(self.repeat):
            # TODO: 生成了新的状态，计算动作仍然采用原来的rnn_state、mask，这样会不会有问题（对于计算动作来说，系统状态与rnn_state有没有绑定的关系）
            new_obs = self.sample_new_obs(obs)
            
            tmp_loss = self.calc_loss(
                policy=policy, 
                action_dist_probs=action_dist_probs, 
                obs=new_obs, 
                rnn_states_actor=rnn_states_actor,
                masks=masks)
            
            loss_list.append(tmp_loss)

        loss_list = torch.stack(loss_list, dim=0)
        return loss_list.mean()
    
    def sample_new_obs(self, obs: np.ndarray) -> np.ndarray:
        """

        Args:
            obs (np.ndarray): (batch_size, observation_space.shape)

        Returns:
            _type_: _description_
        """
        obs_tensor = check(obs)
        return torch.normal(obs_tensor, torch.ones_like(obs_tensor)*self.local_for_sample_new_obs).numpy()

