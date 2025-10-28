# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from autogen import OpenAIWrapper
from secgym.agents.agent_utils import sql_parser, msging, call_llm, call_llm_foundry, update_model_usage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# from tenacity import retry, wait_fixed

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

R1_PROMPT = """You are a security analyst. 
You need to interact with a database with MYSQL queries to collect info and then answer a given security question.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

In you response, you should give your thoughts and actions. The reasoning process and action are enclosed within <think> </think> and <answer> </answer> tags,
You should ONLY have ONE action per response in the <answer> </answer> block, it can be one of the following based on your reasoning:
(1) <answer>execute[<your sql query>]</answer>, which executes the SQL query. For example, execute[DESCRIBE table_name]. You should give sql queries to explore the schema and acquire information.
(2) <answer>submit[<your answer>]</answer>, which submits the final answer to the question. When you believe you have enough information to answer the question, you can submit your answer.

Please do not do excessive reasoning. Briefly reason about the current situation and then give your action quickly. DO NOT make assumptions about the data that are not observed in the logs. Be conside and precice in your response.
"""
 

class BaselineAgent:
    def __init__(self,
                 config_list,
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

        if "o4" in config_list[0]['model']:
            self.temperature = 1
        for i in range(len(config_list)):
            if  config_list[i].get('api_type') is None:
                config_list[i]['api_type'] = "openai"

        if config_list[0].get('api_type') is None:
            pass
        elif "ai_foundry" in config_list[0].get('api_type'):
            from secgym.config_key import api_key
            self.client = ChatCompletionsClient(
            endpoint= config_list[0]['endpoint'],
            credential=AzureKeyCredential(api_key),
            seed =self.cache_seed
            )
        else:
            self.client = OpenAIWrapper(config_list=config_list, cache_seed=cache_seed)
        
        sys_prompt = BASE_PROMPT
        if "o1" in config_list[0]['model'] or "o3" in config_list[0]['model'] or "r1" in config_list[0]['model'] or "R1" in config_list[0]['model'] or "o4" in config_list[0]['model'] or "meta-llama" in config_list[0]['model'] :
            sys_prompt = O1_PROMPT
        self.messages = [{"role": "system", "content": sys_prompt}]
        if "r1" in config_list[0]['model'] or "R1" in self.config_list[0]['model'] or "qwen3" in self.config_list[0]['model']:
            self.messages = [{"role": "system", "content": R1_PROMPT}]  # no system prompt for deepseek
            print("Deepseek model, no system prompt")

        self.max_steps = max_steps
        self.submit_summary = submit_summary
        self.step_count = 0
        self.retry_num = retry_num
        self.retry_wait_time = retry_wait_time
        self.totoal_usage = {}
        
    @property
    def name(self):
        return "BaselineAgent"

    def _call_llm(self, messages):
        # print(f"Messages: {self.config_list[0]}")
        if "ai_foundry" in self.config_list[0]['api_type']:
            response = call_llm_foundry(
                client=self.client, 
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature,
                stop=['</answer>'],
            )
            update_model_usage(self.totoal_usage, model_name=response.model, usage_dict=response.usage.as_dict())
        else:
                    # if "azure" in self.config_list[0]['api_type']:
            response = call_llm(
                client=self.client, 
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature
            )
            update_model_usage(self.totoal_usage, model_name=response.model, usage_dict=response.usage.model_dump())
        print(response)
        return response.choices[0].message.content
        
    def act(self, observation: str):
        if ("r1" in self.config_list[0]['model' or "R1" in self.config_list[0]['model']] or "qwen3" in self.config_list[0]['model']) and len(self.messages) == 0:
            self._add_message(observation, role="user")
        else:
            self._add_message(observation, role="user")
        response = self._call_llm(messages=self.messages)
        print(response)

        if self.step_count >= self.max_steps-1 and self.submit_summary:
            summary_prompt = "You have reached maximum number of steps. Please summarize your findings of key information, and sumbit them."
            self._add_message(summary_prompt, role="system")

        split_str = "\nAction:"
        if "r1" in self.config_list[0]['model'] or "R1" in self.config_list[0]['model'] or "qwen3" in self.config_list[0]['model']:
            split_str = "<answer>"

        if "**Action:**" in response:
            split_str = "\n**Action:**"
        try:
            thought, action = response.strip().split(split_str)
            action = action.replace("<answer>", "").replace("</answer>", "")
            self._add_message(response.strip(), role="assistant")
        except:
            print("\nRetry Split Action:")
            thought = response.strip()
            action = self._call_llm(self.messages + [msging(f"{thought}\nAction:")])
            print(action)
            action = action.strip()
            action = action.replace("<answer>", "").replace("</answer>", "")
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
            "usage_summary": self.totoal_usage,
        }
    
    def _add_message(self, msg: str, role: str="user"):
        self.messages.append(msging(msg, role))

    def reset(self, change_seed=True):
        if change_seed:
            self.cache_seed += 1
        if self.config_list[0].get('api_type') is None:
            pass
        elif "ai_foundry" in self.config_list[0]['api_type']:
            from secgym.config_key import api_key
            self.client = ChatCompletionsClient(
            endpoint= self.config_list[0]['endpoint'],
            credential=AzureKeyCredential(api_key),
            seed =self.cache_seed
            )
        elif "azure" in self.config_list[0]['api_type']:
            self.client = OpenAIWrapper(config_list=self.config_list, cache_seed=self.cache_seed)

        self.step_count = 0
        sys_prompt = BASE_PROMPT
        if "o1" in self.config_list[0]['model'] or "o3" in self.config_list[0]['model'] or "o4" in self.config_list[0]['model'] or "meta-llama" in self.config_list[0]['model'] or "qwen3" in self.config_list[0]['model']:
            sys_prompt = O1_PROMPT
        elif "r1" in self.config_list[0]['model'] or "R1" in self.config_list[0]['model'] or "qwen3" in self.config_list[0]['model']:
            sys_prompt = R1_PROMPT
        self.messages = [{"role": "system", "content": sys_prompt}]
        # if "r1" in self.config_list[0]['model']:
        #     self.messages = []  # no system prompt for deepseek
        #     print("Deepseek model, no system prompt")
        self.totoal_usage = {}
