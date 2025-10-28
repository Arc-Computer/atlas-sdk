

# 1. run evaluation again (bugged)
#python run_evaluation.py

# --------------------------------------------
# prompt sauce, 1 trial, step 15, gpt-4o-mini
#python run_exp.py --agent prompt_sauce --cache_seed 34 --model 4o-mini --num_trials 1 (done)
# run but use 3 trials, make sure use the same cache_seed as above
# python run_exp.py --agent prompt_sauce --cache_seed 34 --model 4o-mini --num_trials 3 (done)

# --------------------------------------------
# prompt sauce, 1 trial, step 15, gpt-4o
# python run_exp.py --agent prompt_sauce --cache_seed 345 --model gpt-4o --num_trials 1 (done)
# run but use 3 trials, make sure use the same cache_seed as above
# python run_exp.py --agent prompt_sauce --cache_seed 345 --model gpt-4o --num_trials 3 (done)


# --------------------------------------------
# gpt-4o run react for 3 trials, keep the same cache_seed as the existing one
# python run_exp.py --agent react --cache_seed 121 --model gpt-4o --num_trials 3 (done)
# 4o-mini run react for 3 trials, keep the same cache_seed as the existing one
# python run_exp.py --agent react --cache_seed 124 --model 4o-mini --num_trials 3 (done)


# --------------------------------------------

# # # prompt sauce reflexion, 3 trials, 4o-mini, rerun with 15 steps
# python run_exp.py --agent ps_reflexion --cache_seed 234 --model 4o-mini --num_trials 3 (done)
# # # prompt sauce reflexion, 3 trials, 4o, rerun with 15 steps
# python run_exp.py --agent ps_reflexion --cache_seed 234 --model gpt-4o --num_trials 3 (done)


# # --------------------------------------------
# # react reflexion, 3 trials, 4o-mini, rerun with 15 steps
# python run_exp.py --agent react_reflexion --cache_seed 378 --model 4o-mini --num_trials 3 (done)
# # # react reflexion, 3 trials, 4o, rerun with 15 steps
# python run_exp.py --agent react_reflexion --cache_seed 378 --model gpt-4o --num_trials 3 (done)

# --------------------------------------------
# TODO: run train set with react reflexion
# python run_exp.py --agent react_reflexion --cache_seed 349 --model gpt-4o --num_trials 3 --split train (done)






