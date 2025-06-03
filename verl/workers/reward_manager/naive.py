# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk, halueval, asqa, hotpot
import torch
import nltk
import re
import numpy as np


def _select_rm_score_fn(data_source):
    if data_source == 'GSM8K':
        return gsm8k.compute_score
    elif data_source == 'MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    elif data_source == 'HaluEval':
        return halueval.compute_score
    elif data_source == 'ASQA':
        return asqa.compute_score
    elif data_source == 'HotpotQA' or data_source == '2WikiMultiHopQA':
        return hotpot.compute_score
    else:
        raise NotImplementedError


class NaiveRewardManager:
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        answer_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            compute_score_fn = _select_rm_score_fn(data_source)
            format_score, answer_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            print("\n" + "-" * 80)
            print(f" Final Score ".center(80, '-'))
            print(f"  Format score: {format_score}")
            print(f"  Answer score: {answer_score}")

            total_score = answer_score
            print(f"  Total: {total_score}")
            print("=" * 80 + "\n")

            reward_tensor[i, valid_response_length - 1] = total_score

            format_reward_tensor[i] = format_score
            answer_reward_tensor[i] = answer_score

        return reward_tensor, format_reward_tensor, answer_reward_tensor
