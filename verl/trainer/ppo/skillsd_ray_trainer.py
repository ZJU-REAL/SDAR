"""
SkillSD (Skill-based Self-Distillation) Trainer.

Extends RLSDRayTrainer. Unlike RLSD which replaces advantages with token-level
teacher-weighted advantages, SkillSD keeps the original GRPO advantages and
adds an auxiliary SDL loss computed inside the actor's update_policy().
"""

from pprint import pprint

import json
import os

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class SkillSDRayTrainer(RLSDRayTrainer):
    """
    SkillSD trainer that extends RLSDRayTrainer.

    Keeps the original GRPO advantages unchanged and passes teacher_log_probs
    to the actor, where the SDL loss is computed inside update_policy().
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        skillsd_cfg = self.config.algorithm.get("skillsd", {})
        self.sdl_lambda = skillsd_cfg.get("sdl_lambda", 0.1)
        self.sdl_warmdown_steps = skillsd_cfg.get("warmdown_steps", -1)

    def _get_sdl_lambda(self, step: int) -> float:
        if self.sdl_warmdown_steps <= 0:
            return self.sdl_lambda
        if step >= self.sdl_warmdown_steps:
            return 0.0
        return self.sdl_lambda * (1.0 - step / self.sdl_warmdown_steps)

    def fit(self):
        """
        The training loop of SkillSD. Identical to RLSD except:
        - Advantages are NOT replaced with token-level advantages
        - teacher_log_probs are passed through to the actor for SDL loss
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                if "task_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("task_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- SkillSD: Teacher forward pass (same as RLSD) ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                invalid_action_penalty_coef_by_task=self.config.actor_rollout_ref.actor.get(
                                    "invalid_action_penalty_coef_by_task", None
                                ),
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- SkillSD: Do NOT replace advantages ----
                        # Advantages stay as standard GRPO sequence-level advantages.
                        # The SDL loss is computed inside dp_actor.update_policy().

                        # Log teacher-student gap metrics
                        response_mask = batch.batch["response_mask"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        current_sdl_lambda = self._get_sdl_lambda(self.global_steps)
                        metrics["skillsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["skillsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["skillsd/sdl_lambda"] = current_sdl_lambda

                        # Save per-token gap data if SAVE_SDAR_DEBUG=1, at test_freq interval
                        if os.environ.get("SAVE_SDAR_DEBUG", "0") == "1" and \
                                self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                            save_dir = os.environ.get(
                                "SAVE_SDAR_DEBUG_DIR",
                                "outputs/sdar_debug"
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                            bs = response_mask.shape[0]
                            turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                            traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                            episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                            episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                            response_ids = batch.batch["responses"]

                            with open(save_path, "w") as f:
                                for i in range(bs):
                                    mask_i = response_mask[i].bool()
                                    valid_count = mask_i.sum().item()
                                    if valid_count == 0:
                                        continue
                                    token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                    tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                    gaps_i = delta_t[i][mask_i].cpu().tolist()
                                    teacher_lps_i = teacher_lp[i][mask_i].cpu().tolist()
                                    student_lps_i = student_log_probs[i][mask_i].cpu().tolist()

                                    record = {
                                        "global_step": self.global_steps,
                                        "sample_idx": i,
                                        "turn_step": int(turn_steps[i]),
                                        "traj_uid": str(traj_uids[i]),
                                        "episode_reward": float(episode_rewards_arr[i]),
                                        "episode_length": float(episode_lengths_arr[i]),
                                        "tokens": tokens_i,
                                        "token_ids": token_ids_i,
                                        "gaps": gaps_i,
                                        "teacher_log_probs": teacher_lps_i,
                                        "student_log_probs": student_lps_i,
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[SDAR Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
