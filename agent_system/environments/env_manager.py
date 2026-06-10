# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def _normalize_multitask_name(task_name: str) -> str:
    task_name = str(task_name).lower()
    if "alfworld" in task_name:
        return "alfworld"
    if "webshop" in task_name:
        return "webshop"
    if "search" in task_name:
        return "search"
    raise ValueError(f"Unsupported multitask task_name: {task_name}")


def _plain_container(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


class MultiTaskEnvironmentManager(EnvironmentManagerBase):
    """Route a mixed batch to existing task-specific environment managers."""

    def __init__(self, managers: Dict[str, EnvironmentManagerBase], task_max_steps: Dict[str, int], config):
        self.managers = managers
        self.task_max_steps = {task: int(task_max_steps[task]) for task in managers}
        self.config = config
        self._task_indices = {}
        self._task_steps = {}
        self._task_done = {}
        self._last_obs_by_task = {}
        self._last_infos_by_task = {}

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if kwargs is None:
            raise ValueError("multitask environment requires env_kwargs with task_name for every sample.")
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()

        task_to_items = defaultdict(list)
        for idx, item_kwargs in enumerate(kwargs):
            if item_kwargs is None or "task_name" not in item_kwargs:
                raise ValueError("Every multitask env_kwargs entry must contain task_name.")
            task = _normalize_multitask_name(item_kwargs["task_name"])
            if task not in self.managers:
                raise ValueError(f"No environment manager configured for task_name={task}.")
            task_to_items[task].append((idx, item_kwargs))

        self._task_indices = {}
        self._task_steps = {}
        self._task_done = {}
        task_obs = {}
        task_infos = {}

        for task, items in task_to_items.items():
            indices, task_kwargs = zip(*items)
            self._task_indices[task] = list(indices)
            obs, infos = self.managers[task].reset(list(task_kwargs))
            for info in infos:
                info["task_name"] = task
            self._task_steps[task] = 0
            self._task_done[task] = np.zeros(len(indices), dtype=bool)
            self._last_obs_by_task[task] = obs
            self._last_infos_by_task[task] = infos
            task_obs[task] = obs
            task_infos[task] = infos

        observations = self._merge_observations(task_obs, len(kwargs))
        infos = self._merge_infos(task_infos, len(kwargs))
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self._task_indices:
            raise RuntimeError("MultiTaskEnvironmentManager.step called before reset.")

        task_obs = {}
        task_rewards = {}
        task_dones = {}
        task_infos = {}

        for task, indices in self._task_indices.items():
            if self._task_done[task].all() or self._task_steps[task] >= self.task_max_steps[task]:
                task_obs[task] = self._last_obs_by_task[task]
                task_rewards[task] = np.zeros(len(indices), dtype=np.float32)
                task_dones[task] = np.ones(len(indices), dtype=bool)
                task_infos[task] = self._done_infos(task)
                continue

            actions = [text_actions[idx] for idx in indices]
            obs, rewards, dones, infos = self.managers[task].step(actions)
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones).reshape(-1).astype(bool)

            self._task_steps[task] += 1
            if self._task_steps[task] >= self.task_max_steps[task]:
                dones = np.ones(len(indices), dtype=bool)

            for info in infos:
                info["task_name"] = task

            self._task_done[task] = np.logical_or(self._task_done[task], dones)
            self._last_obs_by_task[task] = obs
            self._last_infos_by_task[task] = infos

            task_obs[task] = obs
            task_rewards[task] = rewards
            task_dones[task] = dones
            task_infos[task] = infos

        observations = self._merge_observations(task_obs, len(text_actions))
        rewards = self._merge_arrays(task_rewards, len(text_actions), dtype=np.float32)
        dones = self._merge_arrays(task_dones, len(text_actions), dtype=bool)
        infos = self._merge_infos(task_infos, len(text_actions))
        return observations, rewards, dones, infos

    def _done_infos(self, task: str) -> List[Dict]:
        infos = []
        for info in self._last_infos_by_task[task]:
            done_info = dict(info)
            done_info["task_name"] = task
            done_info["is_action_valid"] = to_numpy(True)
            infos.append(done_info)
        return infos

    def _merge_observations(self, task_obs: Dict[str, Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        keys = set()
        for obs in task_obs.values():
            keys.update(obs.keys())

        merged = {}
        for key in keys:
            values = [None] * batch_size
            has_values = False
            for task, obs in task_obs.items():
                obs_values = obs.get(key)
                if obs_values is None:
                    continue
                has_values = True
                for idx, value in zip(self._task_indices[task], obs_values):
                    values[idx] = value
            merged[key] = values if has_values else None
        return merged

    def _merge_infos(self, task_infos: Dict[str, List[Dict]], batch_size: int) -> List[Dict]:
        merged = [None] * batch_size
        for task, infos in task_infos.items():
            for idx, info in zip(self._task_indices[task], infos):
                merged[idx] = info
        return merged

    def _merge_arrays(self, task_values: Dict[str, np.ndarray], batch_size: int, dtype) -> np.ndarray:
        merged = np.zeros(batch_size, dtype=dtype)
        for task, values in task_values.items():
            for idx, value in zip(self._task_indices[task], values):
                merged[idx] = value
        return merged

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        success = defaultdict(list)

        for task, indices in self._task_indices.items():
            task_total_infos = [total_infos[idx] for idx in indices]
            task_total_batch_list = [total_batch_list[idx] for idx in indices]
            task_success = self.managers[task].success_evaluator(
                total_infos=task_total_infos,
                total_batch_list=task_total_batch_list,
                episode_rewards=kwargs.get("episode_rewards"),
                episode_lengths=kwargs.get("episode_lengths"),
            )
            success["success_rate"].extend(task_success["success_rate"].tolist())
            success[f"{task}_success_rate"].extend(task_success["success_rate"].tolist())
            for key, value in task_success.items():
                if key == "success_rate":
                    continue
                success[f"{task}_{key}"].extend(value.tolist())

        return {key: np.array(value) for key, value in success.items()}

    def close(self) -> None:
        for manager in self.managers.values():
            manager.close()


def _get_multitask_tasks(config) -> List[str]:
    multitask_cfg = config.env.get("multitask", {})
    tasks = _plain_container(multitask_cfg.get("tasks", ["alfworld", "search", "webshop"]))
    return [_normalize_multitask_name(task) for task in tasks]


def _get_multitask_task_max_steps(config, tasks: List[str]) -> Dict[str, int]:
    defaults = {"alfworld": 50, "search": 4, "webshop": 15}
    multitask_cfg = config.env.get("multitask", {})
    configured = _plain_container(multitask_cfg.get("max_steps", {}))
    if configured:
        defaults.update({task: int(value) for task, value in configured.items()})
    return {task: defaults[task] for task in tasks}


def _get_multitask_per_task_batch_size(config, tasks: List[str], is_train: bool) -> int:
    if is_train:
        data_task_balance = config.data.get("task_balance", {})
        per_task_batch_size = data_task_balance.get("per_task_batch_size", None)
        total_batch_size = config.data.train_batch_size
    else:
        multitask_cfg = config.env.get("multitask", {})
        per_task_batch_size = multitask_cfg.get("val_per_task_batch_size", None)
        total_batch_size = config.data.val_batch_size

    if per_task_batch_size is None:
        if total_batch_size is None:
            raise ValueError("multitask val_batch_size must be set when val_per_task_batch_size is not configured.")
        if int(total_batch_size) % len(tasks) != 0:
            raise ValueError(f"multitask batch size {total_batch_size} is not divisible by {len(tasks)} tasks.")
        per_task_batch_size = int(total_batch_size) // len(tasks)
    return int(per_task_batch_size)


def _copy_config_for_task(config, env_name: str, max_steps: int):
    task_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    task_config.env.env_name = env_name
    task_config.env.max_steps = max_steps
    return task_config


def _build_multitask_manager(config, tasks: List[str], task_max_steps: Dict[str, int], per_task_batch_size: int, group_n: int, is_train: bool, seed: int, resources_per_worker: Dict):
    managers = {}

    for task in tasks:
        if task == "alfworld":
            from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection

            alf_config_path = os.path.join(os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml")
            env_kwargs = {"eval_dataset": config.env.alfworld.eval_dataset}
            task_config = _copy_config_for_task(config, "alfworld/AlfredTWEnv", task_max_steps[task])
            _envs = build_alfworld_envs(
                alf_config_path,
                seed,
                per_task_batch_size,
                group_n,
                resources_per_worker=resources_per_worker,
                is_train=is_train,
                env_kwargs=env_kwargs,
            )
            managers[task] = AlfWorldEnvironmentManager(_envs, partial(alfworld_projection), task_config)
        elif task == "search":
            from agent_system.environments.env_package.search import build_search_envs, search_projection

            task_config = _copy_config_for_task(config, "search", task_max_steps[task])
            _envs = build_search_envs(
                seed=seed,
                env_num=per_task_batch_size,
                group_n=group_n,
                is_train=is_train,
                env_config=task_config.env,
            )
            managers[task] = SearchEnvironmentManager(_envs, partial(search_projection), task_config)
        elif task == "webshop":
            from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

            if config.env.webshop.use_small:
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle_1000.json")
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2_1000.json")
            else:
                file_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_shuffle.json")
                attr_path = os.path.join(os.path.dirname(__file__), "env_package/webshop/webshop/data/items_ins_v2.json")
            env_kwargs = {
                "observation_mode": "text",
                "num_products": None,
                "human_goals": config.env.webshop.human_goals,
                "file_path": file_path,
                "attr_path": attr_path,
            }
            task_config = _copy_config_for_task(config, "Webshop", task_max_steps[task])
            _envs = build_webshop_envs(
                seed=seed,
                env_num=per_task_batch_size,
                group_n=group_n,
                is_train=is_train,
                env_kwargs=env_kwargs,
                resources_per_worker=resources_per_worker,
            )
            managers[task] = WebshopEnvironmentManager(_envs, partial(webshop_projection), task_config)
        else:
            raise ValueError(f"Unsupported multitask task: {task}")

    return MultiTaskEnvironmentManager(managers=managers, task_max_steps=task_max_steps, config=config)


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if config.env.env_name.lower() == "multitask":
        tasks = _get_multitask_tasks(config)
        task_max_steps = _get_multitask_task_max_steps(config, tasks)
        train_per_task_batch_size = _get_multitask_per_task_batch_size(config, tasks, is_train=True)
        val_per_task_batch_size = _get_multitask_per_task_batch_size(config, tasks, is_train=False)
        if train_per_task_batch_size * len(tasks) != int(config.data.train_batch_size):
            raise ValueError(
                "multitask train batch mismatch: "
                f"{train_per_task_batch_size} * {len(tasks)} != {config.data.train_batch_size}"
            )
        if val_per_task_batch_size * len(tasks) != int(config.data.val_batch_size):
            raise ValueError(
                "multitask val batch mismatch: "
                f"{val_per_task_batch_size} * {len(tasks)} != {config.data.val_batch_size}"
            )

        envs = _build_multitask_manager(
            config=config,
            tasks=tasks,
            task_max_steps=task_max_steps,
            per_task_batch_size=train_per_task_batch_size,
            group_n=group_n,
            is_train=True,
            seed=config.env.seed,
            resources_per_worker=resources_per_worker,
        )
        val_envs = _build_multitask_manager(
            config=config,
            tasks=tasks,
            task_max_steps=task_max_steps,
            per_task_batch_size=val_per_task_batch_size,
            group_n=1,
            is_train=False,
            seed=config.env.seed + 1000,
            resources_per_worker=resources_per_worker,
        )
        if "webshop" in tasks:
            import time

            time.sleep((train_per_task_batch_size * group_n + val_per_task_batch_size) * 0.1)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
