import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm
from ..utils.loss import ActionTemporalSmoothLossCalculator, ActionSpatialSmoothLossCalculator
import sys


class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        
        # custom loss
        self.use_temporal_action_smooth_loss = args.use_temporal_action_smooth_loss
        self.temporal_action_smooth_loss_calculator = ActionSpatialSmoothLossCalculator()
        self.temporal_action_smooth_loss_coef = args.temporal_action_smooth_loss_coef

        self.use_spatial_action_smooth_loss = args.use_spatial_action_smooth_loss
        self.spatial_action_smooth_loss_calculator = ActionSpatialSmoothLossCalculator(local_for_sample_new_obs=0.01, repeat=1)
        self.spatial_action_smooth_loss_coef = args.spatial_action_smooth_loss_coef

    def ppo_update(self, policy: PPOPolicy, sample, ):
        """
        Args:
            sample:
        """
        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch, \
            next_obs_batch, rnn_next_states_actor_batch, next_masks_batch = sample

        # print("in ppo update:")
        # print(obs_batch.shape)
        # print(next_obs_batch.shape)
        # print(actions_batch.shape)
        # print(obs_batch[0])
        # print(next_obs_batch[0])
        # print(actions_batch[0])
        # sys.exit(0)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, action_dist_probs = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch,
                                                                         return_action_dist_probs=True)

        # Obtain the loss function
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)  # size: (batch_size, 1)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        # TODO：在此处增加smooth loss
        # 根据action_dist_probs计算其它loss
        if self.use_temporal_action_smooth_loss:
            temporal_action_smooth_loss = self.temporal_action_smooth_loss_calculator(
                policy=policy,
                action_dist_probs=action_dist_probs,
                obs=next_obs_batch,
                rnn_states_actor=rnn_next_states_actor_batch,
                masks=next_masks_batch
            )

        if self.use_spatial_action_smooth_loss:
            spatial_action_smooth_loss = self.spatial_action_smooth_loss_calculator(
                policy=policy,
                action_dist_probs=action_dist_probs,
                obs=obs_batch,
                rnn_states_actor=rnn_states_actor_batch,
                masks=masks_batch
            )

        if self.use_temporal_action_smooth_loss and self.use_spatial_action_smooth_loss:
            print(f"PPO loss: {policy_loss}, entropy loss: {policy_entropy_loss}, temporal smooth loss: {temporal_action_smooth_loss}, spatial smooth loss: {spatial_action_smooth_loss}")
            loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef
            + self.temporal_action_smooth_loss_coef * temporal_action_smooth_loss + self.spatial_action_smooth_loss_coef * spatial_action_smooth_loss
        elif not self.use_temporal_action_smooth_loss and not self.use_spatial_action_smooth_loss:
            print(f"PPO loss: {policy_loss}, entropy loss: {policy_entropy_loss}")
            loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef
        else:
            raise NotImplementedError("暂未实现只使用ts或ss！！！")

        # Optimize the loss function
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

