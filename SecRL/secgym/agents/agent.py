# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from autogen import OpenAIWrapper
from secgym.agents.agent_utils import sql_parser, msging, call_llm
from typing import Tuple

class Agent:
    def __init__(self,
                 config_list,
                 cache_seed=41,
                 max_steps=15,
                 temperature=0,
                 retry_num=10,
                 retry_wait_time=5,
                 ):
        self.config_list = config_list
        self.client = OpenAIWrapper(config_list=config_list, cache_seed=cache_seed, temperature=temperature)
        self.messages = []
        self.max_steps = max_steps
        self.step_count = 0

        self.retry_num = retry_num
        self.retry_wait_time = retry_wait_time

    @property
    def name(self):
        raise NotImplementedError("Please implement the name property in your agent class.")

    def _call_llm(self, messages):
        response = call_llm(self.client, messages, self.retry_num, self.retry_wait_time)
        return response.choices[0].message.content
        
    def act(self, observation: str) -> Tuple[str, bool]:
        raise NotImplementedError("Please implement the act method in your agent class.")
    
    def get_logging(self) -> dict:
        return {
            "messages": self.messages,
            "usage_summary": self.client.total_usage_summary,
        }
    
    def _add_message(self, msg: str, role: str="user"):
        self.messages.append(msging(msg, role))

    def reset(self):
        raise NotImplementedError("Please implement the reset method in your agent class.")


