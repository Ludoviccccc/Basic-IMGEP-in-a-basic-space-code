import torch
import torch.nn as nn
from torch import distributions
import torch.nn.functional as F
import sys
sys.path.append("../")

from Simulator8085simCli.compiler import compile_program
from Simulator8085simCli.execute_ import execute

import re
import numpy as np
from functions import make_random_source_code
import os
import matplotlib.pyplot as plt
class Frame(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.N = N
    def f(self,x):
        if x =="NOP":
            out = 1
        elif x=="LHLD":
            out = 2
        elif x=="SHLD":
            out = 3
        else:
            out = -1
        return out
    def padding(self,x):
        out = torch.zeros(self.N)
        out[:len(x)] = torch.Tensor(x)
        return out
    def filter(self,program:str):
        return re.findall(r'\b[A-Z]+\b', program)
    def reverse_filter(self,program:list):
        out = ""
        for c in program:
            out+= c + " 7e87e8"*(c!="HLT" and c!="NOP") +"\n"*(c!="HLT")
        return out
    def program2numbers(self,program):
        program_filtered = self.filter(program)
        return list(map(self.f,program_filtered[:-1]))
    def program2padding(self,program:str):
        return self.padding(self.program2numbers(program))



class Policy(Frame):
    def __init__(self,N):
        super().__init__(N)
        self.N = N
        self.linear1 = nn.Linear(self.N,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64, self.N +3) # -1 pour ne pas modifier la dernière ligne
        self.actv = nn.ReLU()
    def forward(self,x:str, logit = False):
        """
        x: str. program
        logit: if True, outputs the action and the logits: action (dict) , logits (torch.Tensor)
        """
        out = self.linear1(self.program2padding(x))
        #out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        dist1 = distributions.Categorical(F.softmax(out[:-3], dim=0))
        line2change = dist1.sample([1]).squeeze()
        dist2 = distributions.Categorical(F.softmax(out[-3:], dim=0))
        change = dist2.sample([1]).squeeze()
        action = {"line":line2change, "change":change}
        if logit:
            out = action, out
        else:
            out = action
        return out
class Epsilon_greedy_policy:
    def __init__(self,N,epsilon,policy):
        self.N = N
        self.epsilon = epsilon
        self.policy = policy
    def __call__(self,x:str, logit = False):
        if torch.bernoulli(torch.Tensor([self.epsilon])):
            out = {"line":torch.randint(0,self.N,(1,))[0],"change":torch.randint(0,3,(1,))[0]}
        else:
            out = self.policy(x)
        return out
class ValueFunction(Frame):
    def __init__(self,N):
        super().__init__(N)
        self.N = N
        self.actv = nn.ReLU()
        self.linear1 = nn.Linear(self.N,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64, 16)
        self.linear4 = nn.Linear(16,1)
    def forward(self,x:str):
        out = self.linear1(self.program2padding(x))
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        return out


class CodeModification(Frame):
    def __init__(self,N):
        super().__init__(N)
        self.instructions= ["NOP","LHLD", "SHLD"]
    def __call__(self,action:dict, program:str):
        line2change = action["line"]
        change = action["change"]
        #print("line", line2change)
        modified_program = self.filter(program)
        #print("change", change)
        #print("program", modified_program)
        if change==0 and line2change < len(modified_program)-1:
            #print("modif")
            #permutation
            if modified_program[line2change]=="NOP":
                modified_program[line2change] = 'LHLD'
            elif modified_program[line2change]=='LHLD':
                modified_program[line2change] = 'SHLD'
            elif modified_program[line2change] == 'SHLD':
                modified_program[line2change] = 'NOP'
        if change==1:
            if len(modified_program)>0 and line2change<len(modified_program)-1:
                #print("sup")
                #delete the instruction
                modified_program[line2change:] = modified_program[line2change+1:]
        if change==2:
            #print("add")
            #if len(modified_program)<self.N:
            if  len(modified_program)<self.N:
                #add an instruction
                istr = np.random.randint(0,len(self.instructions), (1,)).item()
                modified_program[-1] = self.instructions[istr]
                modified_program.append("HLT")

        #print("program", modified_program)
        return self.reverse_filter(modified_program)

class Env(CodeModification):
    def __init__(self, N, goal_interference):
        super().__init__(N)
        self.N = N
        self.goal_interference = goal_interference
        self.actionRep = torch.eye((self.N)* 3).reshape(self.N,3,(self.N)* 3)
    def __call__(self, action:torch.Tensor, program:str)->dict:
        modified_program = super().__call__(action, program)
        modified_program_signature = execute(compile_program(modified_program))
        reward = self.reward(modified_program_signature)
        return {"modified_program": modified_program, "modified_program_signature":modified_program_signature, "reward": reward}
    def distance(self,modified_program_signature):
        out = 0
        for k in modified_program_signature.keys():
            if k=="latence":
                w = 1e4
            elif k=="full time":
                w = 1e3
            else:
                w = 1
            out += w * (modified_program_signature[k] - self.goal_interference[k])**2
        return out
    def reward(self,modified_program_signature):
        out = -(self.distance(modified_program_signature)>10)
        return out
    def end(self,program):
        signature = execute(compile_program(program))
        return self.distance(signature)>10

class OnlineActorCritic(Env):
    #Inconvienient pour le deep learning car pas de batch pour update les reseaux de neurones
    #un algo synchronized parallel actor critic ne peut pas fonctionner car besoin de pluseurs processeur
    #aynschonous parallel actor-critic non plus
    #On pourrait utiliser un algo batch actor critic dans lequel on laisse les episodes se derouler et on met a jour apres
    #On peut utiliser un buffer et faire un off-policy, online actor critic avec la fonction de valeur etat action
    def __init__(self,N, value_function, optimizerV, policy, optimizerPi,goal_interference,epsilon =0.1, gamma = .9,K = 5):
        super().__init__(N,goal_interference)
        self.N = N
        self.K = 1
        self.optimizerV = optimizerV
        self.optimizerPi = optimizerPi
        self.policy = policy
        self.epsilon = epsilon
        self.epsilon_greedy_policy = Epsilon_greedy_policy(self.N,self.epsilon,self.policy)
        self.value_function = value_function
        self.gamma = gamma
        self.loadpath = "loads"
        self.optpath = "opt"
        self.max_norm = 1.0
    def UpdatePi(self,advantage, state,action):
        _, logits_a = self.policy(state, logit = True)

        logit_flat = torch.mul(logits_a[:-3].unsqueeze(1),logits_a[-3:].unsqueeze(0)).flatten()
        logpi = F.cross_entropy(logit_flat,self.actionRep[action["line"],action["change"]],weight = None, reduction = 'none')        
        negativ_pseudo_loss = torch.mul(logpi,advantage)
        self.optimizerPi.zero_grad()
        negativ_pseudo_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm)
        self.optimizerPi.step()
        return negativ_pseudo_loss
    def UpdateV(self,program, target):
        self.optimizerV.zero_grad()
        loss = F.mse_loss(self.value_function(program), target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_norm)

        self.optimizerV.step()
        return loss
    def train(self,start,n_episodes):
        listLosspi = []
        listLossV = []
        self.epsilon_greedy_policy.epsilon = 1
        nb_iterations = []
        for j in range(start,n_episodes+1):
                k = 0
                self.epsilon_greedy_policy.epsilon*=.95
                if self.epsilon_greedy_policy.epsilon<.1:
                    self.epsilon_greedy_policy.epsilon = .1
                #initial program
                if j%1==0:
                    programInit = make_random_source_code(np.random.randint(40))
                program = programInit
                #signature = execute(compile_program(program))
                #while self.distance(signature)>2 and k<200:
                while self.end(program) and k<400:
                    action = self.epsilon_greedy_policy(program)
                    transition = super().__call__(action, program)
                    targets = transition["reward"] + self.gamma*self.value_function(transition["modified_program"])
                    targets = targets.detach()
                    for l in range(self.K):
                        loss = self.UpdateV(program, targets)
                    #advantage evaluation
                    advantage = transition["reward"] + self.gamma*self.value_function(transition["modified_program"]).squeeze() - self.value_function(program)
                    advantage = advantage.detach()
                    #Policy update
                    NegativPseudoLoss = self.UpdatePi(advantage, program, action) # theta <-- theta + alpha* grad J(theta) using negativpseudoloss for policy pi
                    k+=1
                    program = transition["modified_program"]
                    signature = transition["modified_program_signature"]
                    listLosspi.append(NegativPseudoLoss.item())
                    listLossV.append(loss.item())
                if j%1==0:
                    print(f"episod {j} finished with {k} iterations")
                #if j%1==0:
                    print("episodes", j,f"/{n_episodes}")
                    print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
                    print("Loss Q", torch.mean(torch.Tensor(listLossV)))
                    print("epsilon", self.epsilon_greedy_policy.epsilon)
                    print("len program init",len(self.filter(programInit)))

        
                if j%100==0:
                    torch.save(self.policy.state_dict(), os.path.join(self.loadpath,f"pi_load_{j}.pt"))
                    torch.save(self.value_function.state_dict(), os.path.join(self.loadpath,f"v_load_{j}.pt"))

                    torch.save(self.optimizerPi.state_dict(), os.path.join(self.optpath,f"opt_pi_load_{j}.pt"))
                    torch.save(self.optimizerV.state_dict(), os.path.join(self.optpath,f"opt_q_load_{j}.pt"))
                nb_iterations.append(k)
                if j%10:
                    plt.figure()
                    plt.plot(nb_iterations)
                    plt.title("nombres_iterations")
                    plt.xlabel("episode")
                    plt.savefig("image/nb_iterations")
                    plt.close()

                    plt.figure()
                    plt.plot(listLosspi)
                    plt.title("Pseudo Loss Pi")
                    plt.xlabel("episode")
                    plt.savefig("image/losspi")
                    plt.close()


        return nb_iterations        

