# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Tuple, Any, Callable, Union
from glob import glob
import os
import json
import random
import re
import argparse
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider, AzureCliCredential
token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

# Add argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract insights from experience trajectories')
    parser.add_argument('--max_trials', type=int, default=3,
                        help='Maximum number of trials per question')
    parser.add_argument('--json_path', type=str, default='final_results/ReActReflexionAgent_gpt-4o_c349_alert_level_t0_s15_trial3_train',
                        help='Path to experience JSON files')
    parser.add_argument('--correct_batch_size', type=int, default=4,
                        help='Size of batches for correct examples')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name to use for insight extraction')
    parser.add_argument('--seed', type=int, default=24,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_path', type=str, default='./insights.json',
                        help='Path to save extracted insights')
    parser.add_argument('--correct_output_path', type=str, default='./corrects.jsonl',
                        help='Path to save correct trajectories')
    parser.add_argument('--log_output_path', type=str, default='./logs.json',
                        help='Path to save logs')
    parser.add_argument('--steps_per_question', type=int, default=4,
                        help='Number of update steps per question')
    parser.add_argument('--init_starting_vote', type=int, default=2,
                        help='Initial vote count for new insights')
    parser.add_argument('--use_tools', type=bool, default=True,
                        help='Whether to use tools for insight extraction')
    parser.add_argument('--openai_api_key', type=str, default="EMPTY",
                        help='OpenAI API key')
    parser.add_argument('--openai_api_base', type=str, default="http://localhost:8000/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--correct_incorrect_steps', type=int, default=None,
                        help='Number of correct/incorrect pairs to process (None for all)')
    parser.add_argument('--correct_batch_steps', type=int, default=None,
                        help='Number of correct batches to process (None for all)')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Use parsed arguments instead of hardcoded values
MAX_trials = args.max_trials
JSON_PATH = args.json_path
CORRECT_BATCH_SIZE = args.correct_batch_size
MODEL_NAME = args.model_name
SEED = args.seed
OUTPUT_PATH = args.output_path
CORRECT_OUTPUT_PATH = args.correct_output_path
LOG_OUTPUT_PATH = args.log_output_path
STEPS_PER_QUESTION = args.steps_per_question
INIT_STARTING_VOTE = args.init_starting_vote
USE_TOOLS = args.use_tools
OPENAI_API_KEY = args.openai_api_key
OPENAI_API_BASE = args.openai_api_base
CORRECT_INCORRECT_STEPS = args.correct_incorrect_steps
CORRECT_BATCH_STEPS = args.correct_batch_steps


EXPEL_PAIRED_SYSTEM_MESSAGE = """You will be given two previous task trajectories in which you attempted to solve a cybersecurity question, one successful and one unsuccessful. By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, upvote, downvote, or finish so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action."""
TOOL_EXPEL_PAIRED_SYSTEM_MESSAGE = EXPEL_PAIRED_SYSTEM_MESSAGE + ' Use only one tool at a time.'
EXPEL_CORRECT_BATCH_SYSTEM_MESSAGE = """You will be given successful task trajectories in which you attempted to solve cybersecurity questions. By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, upvote, downvote, or finish so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action."""
TOOL_EXPEL_CORRECT_BATCH_SYSTEM_MESSAGE = EXPEL_CORRECT_BATCH_SYSTEM_MESSAGE + ' Use only one tool at a time.'
EXPEL_USER_MESSAGE = """Trajectories:\n{}\n\nExisting Rules:\n{}"""


OPTIMIZER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "downvote_insight",
            "description": "Downvote a specified insight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_idx": {
                        "type": "integer",
                        "description": "The index of the insight to downvote."
                    }
                },
                "required": ["insight_idx"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "upvote_insight",
            "description": "Upvote a specified insight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_idx": {
                        "type": "integer",
                        "description": "The index of the insight to upvote."
                    },
                    
                },
                "required": ["insight_idx"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_insight",
            "description": "Edit the content of a specified insight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_idx": {
                        "type": "integer",
                        "description": "The index of the insight to edit."
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content for the insight."
                    }
                },
                "required": ["insight_idx", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_insight",
            "description": "Add a new insight to the list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The new insight to add."
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish optimizing the insight set.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


class ActionType(Enum):
    ADD = "add"
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    EDIT = "edit"
    FINISH = "finish"


@dataclass
class Action:
    idx: int = field(default=0)
    type: ActionType = field(default=ActionType.ADD)
    content: str = field(default="")


def parse_tool_response(completion: Dict[str, Any], model: str) -> Tuple[Action, bool]:
    """
    Parse the response from OpenAI's tool calling API.

    Args:
        completion: The completion response from OpenAI API containing tool calls

    Returns:
        A tuple of (tool_name, arguments_dict, status)
    """

    if 'qwen' in model.lower():
        completion = completion.choices[0].message.content
        tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL)
    elif 'llama' in model.lower():
        completion = completion.choices[0].message.content
        tool_calls = re.findall(r'<|python_tag|>(.*?)<|eom_id|>', completion, re.DOTALL)
    elif 'gpt' in model.lower():
        tool_calls = []
        if not completion.choices[0].message.tool_calls:
            return Action(type=ActionType.ADD, content=""), False
        for toolcall in completion.choices[0].message.tool_calls:
            tmp = toolcall.function.dict()
            tmp['arguments'] = json.loads(tmp['arguments'])
            tool_calls.append(tmp)
    else:
        raise ValueError(f"Invalid model: {model}")
    if len(tool_calls) == 0:
        return Action(type=ActionType.Add, content=""), False
    try:
        convert_to_dict = False
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                convert_to_dict = True
                break
        if convert_to_dict:
            tool_calls = [json.loads(tool_call) for tool_call in tool_calls]
    except json.JSONDecodeError:
        return Action(type=ActionType.Add, content=""), False
    # TODO: support multiple tool calls
    tool_call = tool_calls[0]
    try:
        if tool_call['name'] == 'finish':
            return Action(type=ActionType.FINISH), True
        elif tool_call['name'] == 'downvote_insight':
            return Action(type=ActionType.DOWNVOTE, idx=tool_call['arguments']['insight_idx']), True
        elif tool_call['name'] == 'upvote_insight':
            return Action(type=ActionType.UPVOTE, idx=tool_call['arguments']['insight_idx']), True
        elif tool_call['name'] == 'edit_insight':
            return Action(type=ActionType.EDIT, idx=tool_call['arguments']['insight_idx'], content=tool_call['arguments']['content']), True
        elif tool_call['name'] == 'add_insight':
            return Action(type=ActionType.ADD, content=tool_call['arguments']['content']), True
        else:
            return Action(type=ActionType.ADD, content=""), False
    except:
        print(f"Error parsing tool call: {tool_call}, continue")
        return Action(type=ActionType.ADD, content=""), False


def collect_message_dicts(experience_paths: List[str], max_trials: int) -> Tuple[List[Dict[str, str]], List[Tuple[Dict[str, str], Dict[str, str]]], List[Dict[str, str]], List[Dict[str, str]]]:
    correct_messages = []
    correct_incorrect_messages_pairs = []
    correct_tasks = []
    correct_incorrect_tasks = []
    for experience_path in experience_paths:
        if experience_path.split("/")[-1].startswith("env_"):
            continue
        print(experience_path)
        experience_dict = json.load(open(experience_path))
        for node_dict in experience_dict:
            # convert node_dict['trials'] to list, it is originally a dict, using string index as keys
            node_dict_trials = [node_dict['trials'][str(i)] for i in range(len(node_dict['trials']))]
            # case of no incorrects
            if len(node_dict_trials) == 1:
                correct_messages.append(dict(node_dict_trials[0]))
                correct_tasks.append({k: format_context(node_dict['question_dict'][k]) for k in ['question', 'context']})
            # case of no corrects
            elif len(node_dict_trials) == max_trials and node_dict['reward'] == 0:
                continue
            else:
                correct_messsages = node_dict_trials[-1]
                for incorrect_trail in node_dict_trials[:-1]:
                    correct_incorrect_messages_pairs.append((dict(correct_messsages), dict(incorrect_trail)))
                    correct_incorrect_tasks.append({k: format_context(node_dict['question_dict'][k]) for k in ['question', 'context']})
                correct_messages.append(dict(correct_messsages))
                correct_tasks.append({k: format_context(node_dict['question_dict'][k]) for k in ['question', 'context']})
    return (
        correct_messages,
        correct_incorrect_messages_pairs,
        correct_tasks,
        correct_incorrect_tasks,
    )


def update_insights(action: Action, list_insights: List[Tuple[str, int]], starting_vote: int) -> List[Tuple[str, int]]:
    if action.type == ActionType.ADD:
        list_insights.append((action.content, starting_vote))
    elif  action.type == ActionType.DOWNVOTE:
        current_votes = list_insights[action.idx][1] - 1
        # remove this insight if votes reach 0
        if current_votes == 0:
            list_insights.pop(action.idx)
        else:
            list_insights[action.idx] = (
                list_insights[action.idx][0],
                current_votes,
            )
    elif action.type == ActionType.UPVOTE:
        current_votes = list_insights[action.idx][1] + 1
        list_insights[action.idx] = (
            list_insights[action.idx][0],
            current_votes,
        )
    elif action.type == ActionType.EDIT:
        list_insights[action.idx] = (
            action.content,
            list_insights[action.idx][1],
        )
    return list_insights


def check_action_validity(action: Action, list_insights: List[Tuple[str, int]]) -> bool:
    if action.type == ActionType.FINISH:
        return True
    elif action.type == ActionType.ADD:
        return True
    elif action.type == ActionType.DOWNVOTE:
        return action.idx < len(list_insights)
    elif action.type == ActionType.UPVOTE:
        return action.idx < len(list_insights)
    elif action.type == ActionType.EDIT:
        return action.idx < len(list_insights)
    else:
        return False


def update_insight_list(messages: Dict[str, str], llm: Callable, max_steps: str, list_insights: List[Tuple[str, int]], starting_vote: int, generation_kwargs: Dict[str, Any] = {}) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    for i in range(1, max_steps + 1):
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                **generation_kwargs,
            )
        except Exception as e:
            print(f"Error in LLM call: {e}")
            messages.append({'role': 'assistant', 'content': ""})
            messages.append({'role': 'user', 'content': "Your last action was not successfully parsed due to content filtering issue with oai. Please try again."})
            continue
        if completion.choices[0].message.content is None:
            messages.append({'role': 'assistant', 'content': ""})
        else:
            messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
        if USE_TOOLS:
            action, success = parse_tool_response(completion, MODEL_NAME)
        else:
            raise NotImplementedError("Needs to be implemented")
        action_valid = check_action_validity(action, list_insights)
        # if the action is finish, break
        if action.type == ActionType.FINISH:
            break
        if i < max_steps:
            if success and action_valid:
                messages.append(
                    {
                        'role': 'user',
                        'content': f'The action {action.type.value} was completed successfully and made changes to the current list of insights. You have {max_steps-i}/{max_steps} steps left.'
                    }
                )
                # update the list of insights
                list_insights = update_insights(action, list_insights, starting_vote)
            elif success and not action_valid:
                messages.append(
                    {
                        'role': 'user',
                        'content': f'The action {action.type.value} was not valid due to index out of bounds. Please revise and try again or try another action. You have {max_steps-i}/{max_steps} steps left.'
                    }
                )
            elif not success:
                messages.append(
                    {
                        'role': 'user',
                        'content': f'The last action was not successfully parsed, please revise and try again or try another action. You have {max_steps-i}/{max_steps} steps left.'
                    }
                )
    return list_insights, messages


def format_insights(list_insights: List[Tuple[str, int]]) -> str:
    return '\n'.join([f'{i}. ' + list_insights[i][0] for i in range(len(list_insights))])


def format_context(context: Union[str, Dict[str, str]]) -> str:
    if isinstance(context, str):
        return context
    else:
        context_string = ""
        while isinstance(context, dict):
            context_key = list(context.keys())[0]
            context_string += f"{context_key}: {context[context_key]}\n"
            context = context[context_key]
        return context_string

def truncate_user_msg_length(
    messages: List[Dict[str, str]],
    truncate_length: int = 20000,
):
    for i, message in enumerate(messages):
        if message['role'] == 'user':
            message['content'] = message['content'][:truncate_length] + ' ... (Truncated)'
        messages[i] = message
    return messages
    

def format_correct_batch(batch_correct_messages: List[Dict[str, str]], task_dicts: List[Dict[str, str]]) -> str:
    """Given a list of correct messages from different nodes, output a clean format for the LLM to ingest"""
    return_string = "<TRAJECTORIES>\n{}\n</TRAJECTORIES>"
    trajectory_string = ""
    for i, (correct_message, task_dict) in enumerate(zip(batch_correct_messages, task_dicts)):
        context_string = format_context(task_dict['context'])
        # we only take the messages, and since the first and second messages are the system prompt and question, we skip them
        truncated_correct_messages = truncate_user_msg_length(correct_message['messages'][2:])
        trajectory_string += f"<trajectory_{i}>\n<question>{task_dict['question']}</question>\n<context>{context_string}</context>\n{truncated_correct_messages}\n</trajectory_{i}>\n"
    return return_string.format(trajectory_string)


def format_correct_incorrect_pair(correct_messages: Dict[str, str], incorrect_messages: Dict[str, str], task_dict: Dict[str, str]) -> str:
    """Given a pair of success and failed messages from the same node, output a clean format for the LLM to ingest"""
    return_string = "<TRAJECTORIES>\n{}\n{}\n</TRAJECTORIES>"
    trajectory_string = ""
    context_string = format_context(task_dict['context'])
    trajectory_string += f"<question>{task_dict['question']}</question>\n<context>{context_string}</context>\n"
    # we only take the messages, and since the first and second messages are the system prompt and question, we skip them
    truncated_correct_messages = truncate_user_msg_length(correct_messages['messages'][2:])
    truncated_incorrect_messages = truncate_user_msg_length(incorrect_messages['messages'][2:])
    trajectory_string += f"<correct_trajectory>\n{truncated_correct_messages}\n</correct_trajectory>\n"
    trajectory_string += f"<incorrect_trajectory>\n{truncated_incorrect_messages}\n</incorrect_trajectory>\n"
    return return_string.format(trajectory_string, trajectory_string)


def get_batched_correct_dicts(
    list_correct_dicts: List[Dict[str, str]],
    correct_tasks: List[Dict[str, str]],
    batch_size: int,
    seed: int,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    # batch the list of dicts into sizes of random mini batches, do not drop the if its not divisible by batch size
    random.seed(seed)
    random.shuffle(list_correct_dicts)
    batched_correct_dicts = [list_correct_dicts[i:i+batch_size] for i in range(0, len(list_correct_dicts), batch_size)]
    batched_correct_tasks = [correct_tasks[i:i+batch_size] for i in range(0, len(correct_tasks), batch_size)]
    return batched_correct_dicts, batched_correct_tasks


def main():
    # get the experiences from local files, extract pairs and list of correct messages...
    experience_paths = glob(os.path.join(JSON_PATH,'*.json'))
    print(f'Found total {len(experience_paths)} experience dicts...')
    correct_messages, correct_incorrect_messages_pairs, correct_tasks, correct_incorrect_tasks = \
        collect_message_dicts(
            experience_paths=experience_paths,
            max_trials=MAX_trials,
        )
    print(f'Collected {len(correct_messages)} correct messages from different nodes, and {len(correct_incorrect_messages_pairs)} correct/incorrect messages pairs from same node...')

    # shuffle batches
    batched_correct_messages, batched_correct_tasks = get_batched_correct_dicts(
        list_correct_dicts=correct_messages,
        correct_tasks=correct_tasks,
        batch_size=CORRECT_BATCH_SIZE,
        seed=SEED,
    )

    # Shuffle correct_incorrect pairs
    paired_indices = list(range(len(correct_incorrect_messages_pairs)))
    random.seed(SEED)
    random.shuffle(paired_indices)
    correct_incorrect_messages_pairs = [correct_incorrect_messages_pairs[i] for i in paired_indices]
    correct_incorrect_tasks = [correct_incorrect_tasks[i] for i in paired_indices]

    # format all messages to string for LLM ingestion
    list_batched_correct_prompts = [format_correct_batch(batch, task) for batch, task in zip(batched_correct_messages, batched_correct_tasks)]
    list_correct_incorrect_pair_prompts = [format_correct_incorrect_pair(*pair, task) for pair, task in zip(correct_incorrect_messages_pairs, correct_incorrect_tasks)]

    # truncated steps
    if CORRECT_INCORRECT_STEPS is not None:
        list_correct_incorrect_pair_prompts = list_correct_incorrect_pair_prompts[:CORRECT_INCORRECT_STEPS]
        # list_correct_incorrect_tasks = list_correct_incorrect_tasks[:CORRECT_INCORRECT_STEPS]
    if CORRECT_BATCH_STEPS is not None:
        list_batched_correct_prompts = list_batched_correct_prompts[:CORRECT_BATCH_STEPS]

    client = AzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint="https://metabase-aoi-eus2.openai.azure.com",
        azure_ad_token_provider=token_provider,
    )


    if USE_TOOLS:
        expel_paired_system_message = TOOL_EXPEL_PAIRED_SYSTEM_MESSAGE
        expel_correct_batch_system_message = TOOL_EXPEL_CORRECT_BATCH_SYSTEM_MESSAGE
    else:
        expel_paired_system_message = EXPEL_PAIRED_SYSTEM_MESSAGE
        expel_correct_batch_system_message = EXPEL_CORRECT_BATCH_SYSTEM_MESSAGE

    # LLM insights updating
    list_insights = []
    logs = []
    # TODO: we can also add top-p, temperature, etc.
    generation_kwargs = {}
    if USE_TOOLS:
        generation_kwargs.update({'tools': OPTIMIZER_TOOLS, 'tool_choice': 'auto'})
    print('Starting to extract insights for correct/incorrect paired trajectories...')
    for correct_incorrect_pair_prompt in list_correct_incorrect_pair_prompts:
        formatted_insights = format_insights(list_insights)
        messages = [
            {'role': 'system', 'content': expel_paired_system_message},
            {'role': 'user', 'content': EXPEL_USER_MESSAGE.format(correct_incorrect_pair_prompt, formatted_insights)},
        ]
        list_insights, messages = update_insight_list(
            messages=messages,
            llm=client,
            max_steps=STEPS_PER_QUESTION,
            list_insights=list_insights,
            starting_vote=INIT_STARTING_VOTE,
            generation_kwargs=generation_kwargs,
        )
        logs.append(messages)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump([list_insight[0] for list_insight in list_insights], f)
    print('Starting to extract insights for batched correct trajectories...')
    for correct_batch_prompt in list_batched_correct_prompts:
        formatted_insights = format_insights(list_insights)
        messages = [
            {'role': 'system', 'content': expel_correct_batch_system_message},
            {'role': 'user', 'content': EXPEL_USER_MESSAGE.format(correct_batch_prompt, formatted_insights)},
        ]
        list_insights, messages = update_insight_list(
            messages=messages,
            llm=client,
            max_steps=STEPS_PER_QUESTION,
            list_insights=list_insights,
            starting_vote=INIT_STARTING_VOTE,
            generation_kwargs=generation_kwargs,
        )
        logs.append(messages)

    # save insights, correct trajs, logs to local
    with open(OUTPUT_PATH, 'w') as f:
        json.dump([list_insight[0] for list_insight in list_insights], f)
    with open(CORRECT_OUTPUT_PATH, 'w') as f:
        for traj, context in zip(correct_messages, correct_tasks):
            f.write(json.dumps({'value': traj, 'key': context}) + '\n')
    with open(LOG_OUTPUT_PATH, 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    main()
