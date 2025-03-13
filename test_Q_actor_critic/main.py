import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from Simulator8085simCli.compiler import compile_program
from Simulator8085simCli.execute_ import execute

from tqdm import tqdm
import re
from Class import Frame, Policy, Env, OnlineActorCritic, QFunction
import torch.optim as optim
from functions import make_random_source_code
import pickle
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__=="__main__":
    N = 50
    batch_size = 100
    number_episode = 100
    train = True
    pi =Policy(N).to(device)
    Q = QFunction(N).to(device)
    if False:
        program = make_random_source_code(25)
        with open("objectif_inter.pkl","wb") as f:
            pickle.dump(program,f)
    else:
        with open("objectif_inter.pkl","rb") as f:
            program = pickle.load(f)

    env = Env(N = N,goal_interference = execute(compile_program(program))).to(device)
    action =pi(program)
    Dict = env(action, program)

    optimzerQ = optim.Adam(Q.parameters(),lr = 1e-5)
    optimizerPi = optim.Adam(pi.parameters(), lr = 1e-5)

    
    start = 0
    if start>0:
        pi.load_state_dict(torch.load(os.path.join(loadpath, f"pi_load_{start}.pt"),weights_only=True))
        optimizerpi.load_state_dict(torch.load(os.path.join(loadopt,f"opt_pi_load_{start}.pt"),weights_only=True))
        V.load_state_dict(torch.load(os.path.join(loadpath,  f"v_load_{start}.pt"),weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt, f"opt_q_load_{start}.pt"),weights_only=True))

    ac = OnlineActorCritic(N, Q, optimzerQ, pi, optimizerPi,env.goal_interference,maxsize=10000, batch_size = batch_size, gamma=.95)
    if train:
        nb_iterations = ac.train(0,number_episode)
