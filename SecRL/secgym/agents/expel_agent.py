# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from autogen import OpenAIWrapper
from secgym.agents.agent_utils import sql_parser, msging, call_llm
from secgym.agents.expel_train.experience_recall import ExperiencePool
import json
BASE_PROMPT = """You are a security analyst. 
You need to answer a given security question by querying the database.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

Your response should always be a thought-action pair:
Thought: <your reasoning>
Action: <your SQL query>

In Thought, you can analyse and reason about the current situation, 
Action can be one of the following: 
(1) execute[<your query>], which executes the SQL query
(2) submit[<your answer>], which is the final answer to the question
"""


O1_PROMPT = """You are a security analyst. 
You need to answer a given security question by querying the database.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

Your response should always be a thought-action pair:
Thought: <your reasoning>
Action: <your action>

In Thought, you can analyse and reason about the current situation, 
Action can be one of the following: 
(1) execute[<your sql query>], which executes the SQL query. For example, execute[DESCRIBE table_name].
(2) submit[<your answer>], which is the final answer to the question

You should only give one thought-action per response. The action from your response will be executed and the result will be shown to you.
Follow the format "Thought: ....\nAction: ...." exactly.
Do not include any other information in your response. Wait for the response from one action before giving the next thought-action pair. DO NOT make assumptions about the data that are not observed in the logs.
"""

class ExpelAgent:
    def __init__(self,
                 config_list,
                 insight_path,
                 experience_path,
                 cache_seed=41,
                 max_steps=15,
                 submit_summary=False,
                 temperature=0,
                 retry_num=10,
                 retry_wait_time=5,
                 ):
        self.cache_seed = cache_seed
        self.config_list = config_list
        self.temperature = temperature
        self.client = OpenAIWrapper(config_list=config_list, cache_seed=cache_seed)
        if "o1" in config_list[0]['model'] or "o3" in config_list[0]['model'] or "r1" in config_list[0]['model']:
            self.sys_prompt = O1_PROMPT + """\nThe following are some experiences you gather on a similar task of question answering. Use these as references to help you answer the current question.\n"""
        else:
            self.sys_prompt = BASE_PROMPT + """\nThe following are some experiences you gather on a similar task of question answering. Use these as references to help you answer the current question.\n"""
        
        with open(insight_path, 'r') as f:
            self.insights = json.load(f)
        for i, insight in enumerate(self.insights):
            self.sys_prompt += f"{i+1}. {insight}\n"

        self.max_steps = max_steps
        self.submit_summary = submit_summary
        self.step_count = 0
        self.retry_num = retry_num
        self.retry_wait_time = retry_wait_time

        self.experience_pool = ExperiencePool(
            correct_trajectories_path=experience_path,
            embed_type="both",
        )

        self.messages = None
    
    @property
    def name(self):
        return "ExpelAgent"

    def _call_llm(self, messages):
        response = call_llm(
            client=self.client, 
            model=self.config_list[0]['model'],
            messages=messages,
            retry_num=self.retry_num,
            retry_wait_time=self.retry_wait_time,
            temperature=self.temperature
        )
        return response.choices[0].message.content
        
    def act(self, observation: str):
        self._add_message(observation, role="user")
        response = self._call_llm(messages=self.messages)
        print(response)

        if self.step_count >= self.max_steps-1 and self.submit_summary:
            summary_prompt = "You have reached maximum number of steps. Please summarize your findings of key information, and sumbit them."
            self._add_message(summary_prompt, role="system")

        split_str = "\nAction:"
        if "**Action:**" in response:
            split_str = "\n**Action:**"
        try:
            thought, action = response.strip().split(split_str)
            self._add_message(response.strip(), role="assistant")
        except:
            print("\nRetry Split Action:")
            thought = response.strip()
            action = self._call_llm(self.messages + [msging(f"{thought}\nAction:")])
            print(action)
            action = action.strip()
            if not "Thought" in thought:
                thought = f"Thought: {thought}"
            self._add_message(f"{thought}\nAction:{action}", role="assistant")
        
        print("*"*50)

        self.step_count += 1
        # parse the action
        parsed_action, is_code, submit = sql_parser(action)
        
        # submit = True if "submit" in thought.lower() else False
        return parsed_action, submit
    
    def get_logging(self):
        return {
            "messages": self.messages,
            "usage_summary": self.client.total_usage_summary,
        }
    
    def _add_message(self, msg: str, role: str="user"):
        self.messages.append(msging(msg, role))

    def reset(self, change_seed=True, question_dict=None):
        if change_seed:
            self.cache_seed += 1
        self.client = OpenAIWrapper(config_list=self.config_list, cache_seed=self.cache_seed)

        demo_str = self.experience_pool.get_demonstrations({
            "context": question_dict['context'],
            "question": question_dict['question'],
        }, k=1)

        sys_prompt = f"{self.sys_prompt}\n\nHere are some examples:\n{demo_str}\n"
        self.step_count = 0
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.client.clear_usage_summary()
