# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
random.seed(0)
import json

VERY_LONG_RETURN = 40000

def analysis(data:dict, verbose:bool=False, round_cut=-1):
    """Analyze the data and print out the results

    Args:
        data (dict): The data to be analyzed, load an "agent log" json file
        
    """
    if round_cut != -1 and "trials" in data[0]:
        # print("Warning: use round_cut with trial data")
        pass
    total_len = len(data)
    total_reward = 0
    total_round = 0
    
    success_count = 0 # reward == 1
    non_zero_reward_count = 0 # reward > 0
    submit_count = 0

    path_count = {}
    reward_count = {} # reward for each difficulty
    round_count = {}
    eval_error_count = 0

    # cost
    total_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # query efficiency
    empty_result_count = 0
    error_query_count = 0
    query_count = 0
    desc_count = 0
    total_select_count = 0
    return_char_len = 0
    reward_list = []
    char_len_list = []
    avg_char_len_list = []
    very_long_return_count_list = []
    select_error_rate_list = []

    success_non_empty_query_count = 0
    potential_good_example_count = 0

    fail_to_run_count = 0
    max_round = 0
    min_round = 1000

    for k in data:
        if "trials" not in k and k.get("usage_summary") is None:
            print(f"No usage summary, skipping {k['nodes']}")
            fail_to_run_count += 1
            continue

        p = len(k['question_dict']['shortest_alert_path'])
        if p not in path_count:
            path_count[p] = 0
            reward_count[p] = 0
            round_count[p] = 0

        # situation 1: "trials" not in k
        if "trials" not in k:
            if "info" in k:
                submit_count += k['info'].get("submit", 0)
            
            tmp_char_len = 0
            return_count = 0
            very_long_count = 0
            select_error_count = 0
            single_problem_query_count = 0
            for i, m in enumerate(k['messages']):
                content = m['content']
                if m['role'] == "user":
                    if "ProgrammingError" in content or "DataError" in content:
                        error_query_count += 1
                        select_error_count += 1
                    elif content == "[]" :
                        empty_result_count += 1
                    elif i > 1:
                        tmp_char_len += len(content)
                        return_count += 1
                        if len(content) > VERY_LONG_RETURN:
                            very_long_count += 1

                if m['role'] == "assistant":
                    if "Action: execute[" in content:
                        query_count += 1
                        single_problem_query_count += 1

                        if "DESC" in content:
                            desc_count += 1
                        if "SELECT" in content:
                            total_select_count += 1
            return_char_len += tmp_char_len
            char_len_list.append(tmp_char_len)
            avg_char_len_list.append(tmp_char_len / return_count if return_count > 0 else 0)
            very_long_return_count_list.append(very_long_count)
            select_error_rate_list.append(select_error_count / single_problem_query_count if single_problem_query_count > 0 else 0)

            tmp_round = (len(k["messages"]) - 1) // 2
            if round_cut != -1 and tmp_round > round_cut:
                # print(f"Cutting off at round {round_cut}")
                k['reward'] = 0
                tmp_round = round_cut
            total_round += tmp_round
            total_reward += k['reward']
            if k['reward'] > 0:
                non_zero_reward_count += 1
            if k['reward'] == 1:
                success_count += 1
            
            reward_list.append(k['reward'])
            path_count[p] += 1
            reward_count[p] += k['reward']
            total_cost, total_prompt_tokens, total_completion_tokens = add_to_usage(k['usage_summary'], total_cost, total_prompt_tokens, total_completion_tokens)
        else:
            raise NotImplementedError("Not implemented for question dict with trials field, use another function")
            
    if verbose:
        print(f"Average reward: {total_reward}/{total_len} = {round(total_reward/total_len,6)}")
        print(f"Average round: {total_round}/{total_len} = {round(total_round/total_len,6)}")


    return {
        "total_len": total_len,

        # Performance Analysis
        "total_reward": total_reward,
        "success_count": success_count, # reward==1
        "total_round": total_round,
        "non_zero_reward_count": non_zero_reward_count, # for evaluator usefullness
        "submit_count": submit_count,

        # Efficiency / Computational Analysis
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,

        # Query Efficiency
        "empty_result_count": empty_result_count,
        "error_query_count": error_query_count,
        "query_count": query_count,
        "desc_count": desc_count,
        "select_count": total_select_count,
        "return_char_len": return_char_len,
        "select_error_rate_list": select_error_rate_list,
        "very_long_return_count_list": very_long_return_count_list,

        "reward_list": reward_list,
        "char_len_list": char_len_list,
        "avg_char_len_list": avg_char_len_list,

        # Evaluation Analysis
        "eval_error_count": eval_error_count,
        "fail_to_run_count": fail_to_run_count
    }

def analysis_per_level(data: list[dict], verbose: bool = False, round_cut: int = -1):
    """
    Analyze agent‑log dictionaries.

    Parameters
    ----------
    data : list[dict]
        The parsed agent‑log JSON (list‑of‑dicts).
    verbose : bool, optional
        If True, print a short summary (average reward / round). Default False.
    round_cut : int, optional
        If >‑1, clamp the counted rounds to this ceiling. Useful for
        head‑to‑head comparisons when logs have different interaction budgets.
    Returns
    -------
    dict
        Metrics, including per‑difficulty counts that were present in the
        original implementation (path_count, reward_count, round_count).
    """
    if round_cut != -1 and "trials" in data[0]:
        # Users may want to catch this early:
        # warn but continue — a dedicated trials‑aware analysis exists elsewhere.
        pass

    total_len = len(data)
    total_reward = total_round = 0
    success_count = non_zero_reward_count = submit_count = 0

    # Difficulty‑conditioned accumulators
    path_count: dict[int, int] = {}
    reward_count: dict[int, float] = {}
    round_count: dict[int, int] = {}

    # Usage / token cost
    total_cost = total_prompt_tokens = total_completion_tokens = 0

    # Query efficiency
    empty_result_count = error_query_count = query_count = 0

    eval_error_count = fail_to_run_count = 0
    max_round, min_round = 0, 1_000

    for k in data:
        # Skip logs without usable usage_summary (unless they belong to trials)
        if "trials" not in k and k.get("usage_summary") is None:
            print(f"No usage summary, skipping {k.get('nodes', '<unknown>')}")
            fail_to_run_count += 1
            continue
        if "trials" in k:
            raise NotImplementedError(
                "Logs with a 'trials' key should be passed to a trials‑aware "
                "analysis function."
            )

        # Difficulty = shortest path length
        p = len(k["question_dict"]["shortest_alert_path"])
        path_count.setdefault(p, 0)
        reward_count.setdefault(p, 0.0)
        round_count.setdefault(p, 0)

        # Interaction rounds
        tmp_round = (len(k["messages"]) - 1) // 2
        if round_cut != -1 and tmp_round > round_cut:
            k["reward"] = 0
            tmp_round = round_cut

        # Accumulate global metrics
        total_round += tmp_round
        total_reward += k["reward"]
        if k["reward"] > 0:
            non_zero_reward_count += 1
        if k["reward"] == 1:
            success_count += 1

        # Accumulate difficulty‑conditioned metrics
        path_count[p] += 1
        reward_count[p] += k["reward"]
        round_count[p] += tmp_round

        # Cost accounting
        total_cost, total_prompt_tokens, total_completion_tokens = add_to_usage(
            k["usage_summary"], total_cost, total_prompt_tokens, total_completion_tokens
        )

        max_round = max(max_round, tmp_round)
        min_round = min(min_round, tmp_round)

    if verbose:
        print(f"Average reward: {total_reward}/{total_len} = {total_reward/total_len:.6f}")
        print(f"Average round:  {total_round}/{total_len} = {total_round/total_len:.6f}")

    return {
        # Dataset‑level counts
        "total_len": total_len,
        "fail_to_run_count": fail_to_run_count,

        # Performance
        "total_reward": total_reward,
        "success_count": success_count,
        "non_zero_reward_count": non_zero_reward_count,
        "submit_count": submit_count,

        # Efficiency / cost
        "total_round": total_round,
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,

        # Query efficiency
        "empty_result_count": empty_result_count,
        "error_query_count": error_query_count,
        "query_count": query_count,

        # Evaluation
        "eval_error_count": eval_error_count,

        # Per‑difficulty aggregates (added back)
        "path_count": path_count,       # number of questions per difficulty
        "reward_count": reward_count,   # sum of rewards per difficulty
        "round_count": round_count,     # total rounds per difficulty
    }

def analysis_v2(data:dict, verbose:bool=False, round_cut=-1):
    """Analyze the data and print out the results

    Args:
        data (dict): The data to be analyzed, load an "agent log" json file
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        round_cut (int, optional): Maximum round to consider. Defaults to -1 (no cut).
    """
    total_len = len(data)
    total_reward = 0
    total_round = 0
    
    success_count = 0 # reward == 1
    non_zero_reward_count = 0 # reward > 0
    submit_count = 0

    path_count = {}
    reward_count = {} # reward for each difficulty
    round_count = {}
    eval_error_count = 0

    # cost
    total_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # query efficiency
    empty_result_count = 0
    error_query_count = 0
    query_count = 0
    desc_count = 0
    total_select_count = 0
    return_char_len = 0
    select_error_rate_list = []

    reward_list = []
    char_len_list = []
    avg_char_len_list = []
    very_long_return_count_list = []


    success_non_empty_query_count = 0
    potential_good_example_count = 0

    fail_to_run_count = 0
    max_round = 0
    min_round = 1000

    for k in data:
        p = len(k['question_dict']['shortest_alert_path'])
        if p not in path_count:
            path_count[p] = 0
            reward_count[p] = 0
            round_count[p] = 0

        assert "trials" in k, "Expecting trials in question dict"
        assert len(k['trials']) == 1, f"Expecting only one trial, got {len(k['trials'])}"
        
        last_trial = list(k["trials"].values())[-1]
        
        # Apply round cut if specified
        tmp_round = (len(last_trial.get("messages", [])) - 1) // 2
        if round_cut != -1 and tmp_round > round_cut:
            reward = 0  # Set reward to 0 if exceeding round cut
            tmp_round = round_cut
        else:
            reward = k['reward']
        
        total_round += tmp_round

        # 1. count reward
        total_reward += reward
        if reward > 0:
            non_zero_reward_count += 1
        if reward == 1:
            success_count += 1
        path_count[p] += 1
        reward_count[p] += reward
        reward_list.append(reward)
        
        # 2. count usage
        try:
            model = list(last_trial['usage_summary'].keys())[-1]
            total_prompt_tokens += last_trial['usage_summary'][model]['prompt_tokens']
            total_completion_tokens += last_trial['usage_summary'][model]['completion_tokens']
        except Exception as e:
            print(f"Error calculating usage: usage summary {last_trial['usage_summary']}")
        
        # 4. check if submitted, if evaluated correctly
        if "info" in last_trial and last_trial['info'].get("submit"):
            submit_count += 1
            if not (last_trial['info'].get("is_json_success", True) and last_trial['info'].get("is_reflect_success", True)):
                print(f"Eval error for {k['nodes']}", last_trial['info'])
                eval_error_count += 1

        k = last_trial
        tmp_char_len = 0
        return_count = 0
        very_long_count = 0
        select_error_count = 0
        single_problem_query_count = 0
        for i, m in enumerate(k['messages']):
            content = m['content']
            if m['role'] == "user":
                if "ProgrammingError" in content or "DataError" in content:
                    error_query_count += 1
                    select_error_count += 1
                elif content == "[]" :
                    empty_result_count += 1
                elif i > 1:
                    tmp_char_len += len(content)
                    return_count += 1
                    if len(content) > VERY_LONG_RETURN:
                        very_long_count += 1

            if m['role'] == "assistant":
                if "Action: execute[" in content:
                    query_count += 1
                    single_problem_query_count += 1

                    if "DESC" in content:
                        desc_count += 1
                    if "SELECT" in content:
                        total_select_count += 1
        return_char_len += tmp_char_len
        char_len_list.append(tmp_char_len)
        avg_char_len_list.append(tmp_char_len / return_count if return_count > 0 else 0)
        very_long_return_count_list.append(very_long_count)
        select_error_rate_list.append(select_error_count / single_problem_query_count if single_problem_query_count > 0 else 0)

    if verbose:
        print(f"Average reward: {total_reward}/{total_len} = {round(total_reward/total_len,6)}")
        print(f"Average round: {total_round}/{total_len} = {round(total_round/total_len,6)}")

    return {
        "total_len": total_len,

        # Performance Analysis
        "total_reward": total_reward,
        "success_count": success_count, # reward==1
        "non_zero_reward_count": non_zero_reward_count, # for evaluator usefullness
        "submit_count": submit_count,
        
        # Round information
        "total_round": total_round,

        # Efficiency / Computational Analysis
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,

        # Query Efficiency
        "empty_result_count": empty_result_count,
        "error_query_count": error_query_count,
        "query_count": query_count,
        "desc_count": desc_count,
        "select_count": total_select_count,
        "return_char_len": return_char_len,
        "avg_char_len_list": avg_char_len_list,
        "very_long_return_count_list": very_long_return_count_list,
        "select_error_rate_list": select_error_rate_list,

        "reward_list": reward_list,
        "char_len_list": char_len_list,

        # Evaluation Analysis
        "eval_error_count": eval_error_count,
        "fail_to_run_count": fail_to_run_count
    }

def add_to_usage(usage_summary:dict, total_cost=0, total_prompt_tokens=0, total_completion_tokens=0):
    model = list(usage_summary.keys())[-1]
    total_cost += usage_summary[model]['cost']
    total_prompt_tokens += usage_summary[model]['prompt_tokens']
    total_completion_tokens += usage_summary[model]['completion_tokens']
    return total_cost, total_prompt_tokens, total_completion_tokens    

def print_analysis(result_dict:dict, head:str=None):
    print("*"*40)
    if head:
        print(f"{head} analysis")
    problem_count = result_dict['total_len']

    print(f"Average reward: {result_dict['total_reward']}/{problem_count} = {round(result_dict['total_reward']/problem_count,4)}")
    print(f"Success rate: {result_dict['success_count']}/{problem_count} = {round(result_dict['success_count']/problem_count * 100,2)}%")
    
    # Computational Analysis
    print(f"** Computational Analysis:")
    print(f"Average round: {result_dict['total_round']}/{problem_count} = {round(result_dict['total_round']/problem_count,4)}")
    print(f"Average cost: {result_dict['total_cost']}/{problem_count} = {round(result_dict['total_cost']/problem_count,4)}")
    print(f"Average prompt tokens: {result_dict['total_prompt_tokens']}/{problem_count} = {round(result_dict['total_prompt_tokens']/problem_count,4)}")
    print(f"Average completion tokens: {result_dict['total_completion_tokens']}/{problem_count} = {round(result_dict['total_completion_tokens']/problem_count,4)}")

    print("** Query Efficiency: ")
    success_query_count = result_dict['query_count'] - result_dict['error_query_count']
    print(f'Success Query rate: {success_query_count}/{result_dict["query_count"]} = {round(success_query_count/result_dict["query_count"] * 100,2)}%')
    print(f'Success Non-Empty Query rate: {success_query_count-result_dict["empty_result_count"]}/{result_dict["query_count"]} = {round((success_query_count-result_dict["empty_result_count"])/result_dict["query_count"] * 100,2)}%')
    
    print(f"Potential Good Example Count: {result_dict['potential_good_example_count']} / {problem_count}")
    print(f"Fail to run count: {result_dict['fail_to_run_count']}")

def get_correct_problem_ids(log_path, file_template, incident_id):
    a = open(f"{log_path}/{file_template.format(incident_id)}", "r")
    b = json.load(a)
    correct_ids = []
    for k in b:
        if k['reward'] == 1:
            correct_ids.append(k['question_id'])
    return correct_ids



def get_over_leaf_format(log_path, file_folder, version="v1", round_cut=-1):
    file_template = f"{log_path}/{file_folder}" + "/agent_incident_{0}.json"

    total_count = 0
    total_reward = 0
    total_success_count = 0
    total_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    accs_str = ""

    incidents = [5, 34, 38, 39, 55, 134, 166, 322]
    for i in incidents:
        # print(f"Analysis for incident {i}")
        with open(file_template.format(i), "r") as f:
            data = json.load(f)
        if version == "v2":
            result = analysis_v2(data, False)
        else:
            result = analysis(data, False, round_cut)
        # print(result)
        accs_str += "& " + str(round(result['total_reward']/result['total_len'], 3)) + " "

        total_count += result['total_len']
        total_reward += result['total_reward']
        total_success_count += result['success_count']
        total_cost += result['total_cost']
        total_prompt_tokens += result['total_prompt_tokens']
        total_completion_tokens += result['total_completion_tokens']
    
    accs_str += "& " + str(round(total_reward/total_count, 3)) + " "
    print(total_count)
    # accs_str += "& " + str(round(total_cost/total_count, 3)) + " "
    print(accs_str)
    return round(total_reward/total_count, 3)

def get_query_metrics(log_path, file_folder, version="v1", round_cut=-1):
    """
    Calculate total query metrics across all incidents.
    
    Args:
        log_path (str): Base path for logs
        file_folder (str): Folder containing log files
        version (str): Analysis version to use ('v1' or 'v2')
        round_cut (int): Maximum round to consider (-1 for no limit)
        
    Returns:
        dict: Dictionary containing total metrics for queries
    """
    file_template = f"{log_path}/{file_folder}" + "/agent_incident_{0}.json"

    total_metrics = {
        "empty_result_count": 0,
        "error_query_count": 0,
        "query_count": 0,
        "desc_count": 0,
        "select_count": 0,
        "total_reward": 0,
        "return_char_len": 0,
        "total_count": 0,
        "submit_count": 0,
        "reward_list": [],
        "char_len_list": [],
        "avg_char_len_list": [],
        "very_long_return_count_list": [],
        "select_error_rate_list": [],
    }

    incidents = [5, 34, 38, 39, 55, 134, 166, 322]
    for i in incidents:
        with open(file_template.format(i), "r") as f:
            data = json.load(f)
        
        if version == "v2":
            result = analysis_v2(data, False, round_cut)
        else:
            result = analysis(data, False, round_cut)

        total_metrics["total_count"] += result["total_len"]
        total_metrics["total_reward"] += result["total_reward"]
        # Accumulate metrics
        total_metrics["empty_result_count"] += result["empty_result_count"]
        total_metrics["error_query_count"] += result["error_query_count"]
        total_metrics["query_count"] += result["query_count"]
        total_metrics["desc_count"] += result["desc_count"]
        total_metrics["select_count"] += result["select_count"]
        total_metrics["return_char_len"] += result["return_char_len"]
        total_metrics["reward_list"].extend(result["reward_list"])
        total_metrics["char_len_list"].extend(result["char_len_list"])
        total_metrics["avg_char_len_list"].extend(result["avg_char_len_list"])
        total_metrics["very_long_return_count_list"].extend(result["very_long_return_count_list"])
        total_metrics["select_error_rate_list"].extend(result["select_error_rate_list"])
        total_metrics["submit_count"] += result["submit_count"]
    
    total_metrics["avg_reward"] = total_metrics["total_reward"] / total_metrics["total_count"] if total_metrics["total_count"] > 0 else 0
    # Calculate averages per problem
    total_metrics["avg_empty_result"] = total_metrics["empty_result_count"] / total_metrics["total_count"]
    total_metrics["avg_error_query"] = total_metrics["error_query_count"] / total_metrics["total_count"]
    total_metrics["avg_query"] = total_metrics["query_count"] / total_metrics["total_count"]
    total_metrics["avg_desc"] = total_metrics["desc_count"] / total_metrics["total_count"]
    total_metrics["avg_select"] = total_metrics["select_count"] / total_metrics["total_count"]
    total_metrics["select_rate"] = total_metrics["select_count"] / total_metrics["query_count"] if total_metrics["query_count"] > 0 else 0
    total_metrics["avg_return_chars"] = total_metrics["return_char_len"] / total_metrics["total_count"]
    total_metrics["error_query_rate"] = total_metrics["error_query_count"] / total_metrics["query_count"] if total_metrics["query_count"] > 0 else 0
    total_metrics["empty_result_rate"] = total_metrics["empty_result_count"] / total_metrics["query_count"] if total_metrics["query_count"] > 0 else 0
    total_metrics["success_query_rate"] = (total_metrics["query_count"] - total_metrics["error_query_count"]- total_metrics["empty_result_count"]) / total_metrics["query_count"] if total_metrics["query_count"] > 0 else 0
    total_metrics["submit_rate"] = total_metrics["submit_count"] / total_metrics["total_count"] if total_metrics["total_count"] > 0 else 0
    # Calculate query efficiency
    if total_metrics["query_count"] > 0:
        total_metrics["successful_query_rate"] = (total_metrics["query_count"] - total_metrics["error_query_count"]) / total_metrics["query_count"]
        total_metrics["non_empty_query_rate"] = (total_metrics["query_count"] - total_metrics["error_query_count"] - total_metrics["empty_result_count"]) / total_metrics["query_count"]
    else:
        total_metrics["successful_query_rate"] = 0
        total_metrics["non_empty_query_rate"] = 0
        
    return total_metrics
