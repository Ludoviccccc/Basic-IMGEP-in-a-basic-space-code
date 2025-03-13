import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from compiler import compile_program
from Simulator8085simCli.execute_ import execute
from tqdm import tqdm
import re
from Class import Frame, Policy, ValueFunction, Env, OnlineActorCritic
import torch.optim as optim
from functions import make_random_source_code
import pickle

if __name__=="__main__":
    N = 50
    train = True
    pi =Policy(N)
    V = ValueFunction(N)
    if False:
        program = make_random_source_code(25)
        with open("objectif_inter.pkl","wb") as f:
            pickle.dump(program,f)
    else:
        with open("objectif_inter.pkl","rb") as f:
            program = pickle.load(f)

    env = Env(N = N,goal_interference = execute(compile_program(program)))
    action =pi(program)
    Dict = env(action, program)

    print("program" ,program)
    print("goal", env.goal_interference)
    print("action", action)
    print("Modified program \n", Dict["modified_program"])
    print("Modified program signature \n", Dict["modified_program_signature"])
    print("reward \n", Dict["reward"])

    optimzerV = optim.Adam(V.parameters(),lr = 1e-4)
    optimizerPi = optim.Adam(pi.parameters(), lr = 1e-4)
    
    start = 0
    if start>0:
        pi.load_state_dict(torch.load(os.path.join(loadpath, f"pi_load_{start}.pt"),weights_only=True))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,f"opt_pi_load_{start}.pt"),weights_only=True))
        V.load_state_dict(torch.load(os.path.join(loadpath,  f"v_load_{start}.pt"),weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt, f"opt_q_load_{start}.pt"),weights_only=True))

    ac = OnlineActorCritic(N, V, optimzerV, pi, optimizerPi,env.goal_interference)
    if train:
        nb_iterations = ac.train(0,1000)
