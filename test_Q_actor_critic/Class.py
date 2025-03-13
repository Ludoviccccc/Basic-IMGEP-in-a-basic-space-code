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
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        out[:len(x)] = torch.Tensor(x).to(device)
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
        return self.padding(self.program2numbers(program)).reshape((1,-1)).to(device)
    def program2paddingvect(self,programs:list[str]):
        out = pad_sequence(list(map(self.program2padding,programs))).to(device)
        #print("out shape" ,out.shape)
        #print("out ndim", out.ndim)
        if len(programs)>1:
            return out.squeeze()
        else:
            return out.reshape((1,-1))



class Policy(Frame):
    def __init__(self,N):
        super().__init__(N)
        self.N = N
        self.linear1 = nn.Linear(self.N,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,64)
        self.linear4 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64, self.N +3) # -1 pour ne pas modifier la dernière ligne
        self.actv = nn.ReLU()
    def forward(self,x:str, logit = False):
        """
        x: str. program
        logit: if True, outputs the action and the logits: action (dict) , logits (torch.Tensor)
        """
        if type(x)==str:
            out = self.linear1(self.program2padding(x))
        elif type(x)==list:
            out = self.linear1(self.program2paddingvect(x))
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        out = self.actv(out)
        out = self.linear5(out)
        dist1 = distributions.Categorical(F.softmax(out[:,:-3], dim=1))
        line2change = dist1.sample([1]).flatten().to(device)
        dist2 = distributions.Categorical(F.softmax(out[:,-3:], dim=1))
        change = dist2.sample([1]).flatten().to(device)
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
    def __call__(self,programs:list[str], logit = False):
        actions = self.policy(programs)
        list_line = actions["line"]
        list_change = actions["change"]
        eps = torch.bernoulli(torch.ones(len(programs))*self.epsilon).to(torch.int64).to(device)
        out_list_line = list_line*(1-eps)+eps*torch.randint(0,self.N,(len(programs),)).to(device)
        out_list_change = list_change*(1-eps)+eps*torch.randint(0,3,(len(programs),)).to(device)
        return {"line":out_list_line, "change":out_list_change}
class QFunction(Frame):
    def __init__(self,N):
        super().__init__(N)
        self.N = N
        self.actv = nn.ReLU()
        self.linear1 = nn.Linear(self.N + self.N + 3,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64, 16)
        self.linear4 = nn.Linear(16,1)
        #self.changereptab = torch.eye(3)
    def changerep(self, change):
        f = lambda x:np.eye(3)[x]
        return torch.Tensor(np.array(list(map(f,change))))
    def linerep(self, line):
        f = lambda x:np.eye(self.N)[x]
        return torch.Tensor(np.array(list(map(f,line))))
    def repaction(self,action):
        #print(self.linerep(action["line"]))
        return torch.cat((self.linerep(action["line"]),self.changerep(action["change"])), dim=1)
    def forward(self,program:list[str],action:dict[list]):
        out = self.linear1(torch.cat((self.program2paddingvect(program),self.repaction(action).to(device)), dim = 1))
        out = self.actv(out)
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
        modified_program = self.filter(program)
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
                w = 1e5
            elif k=="full time":
                w = 1e4
            else:
                w = 1
            out += w * (modified_program_signature[k] - self.goal_interference[k])**2
        return out
    def reward(self,modified_program_signature):
        out = -1.0*(self.distance(modified_program_signature))
        return out
    def end(self,program):
        signature = execute(compile_program(program))
        return self.distance(signature)>1.0
    def actionRepvect(self,action):
        #print("line", action["line"])
        #print("change",action["change"])
        return pad_sequence([self.actionRep[action["line"][j],action["change"][j]] for j in range(len(action["line"]))]).permute(1,0).to(device)

class OnlineActorCritic(Env):
    #Inconvienient pour le deep learning car pas de batch pour update les reseaux de neurones
    #un algo synchronized parallel actor critic ne peut pas fonctionner car besoin de pluseurs processeur
    #aynschonous parallel actor-critic non plus
    #On pourrait utiliser un algo batch actor critic dans lequel on laisse les episodes se derouler et on met a jour apres
    #On peut utiliser un buffer et faire un off-policy, online actor critic avec la fonction de valeur etat action
    def __init__(self,N,
            q_function,
            optimizerQ,
            policy,
            optimizerPi,
            goal_interference,
            epsilon =0.1,
            gamma = .9,
            K = 5,
            maxsize = 10,
            batch_size  = 10):
        super().__init__(N,goal_interference)
        self.N = N
        self.K = 5
        self.optimizerQ = optimizerQ
        self.optimizerPi = optimizerPi
        self.policy = policy
        self.epsilon = epsilon
        self.epsilon_greedy_policy = Epsilon_greedy_policy(self.N,self.epsilon,self.policy)
        self.q_function = q_function
        self.gamma = gamma
        self.loadpath = "loads"
        self.optpath = "opt"
        self.buffer = Buffer(maxsize)
        self.batch_size = batch_size
        self.max_norm = .5

    def UpdatePi(self,samp):
        state = samp["program"]
        api, logits_a = self.policy(state, logit = True)
        advantage = self.q_function(state,api)
        advantage = advantage.detach()
        logit_flat = torch.einsum('bi,bj->bij', logits_a[:,:-3], logits_a[:,-3:]).flatten(1,-1)
        action_rep = self.actionRepvect(api)
        logpi = F.cross_entropy(logit_flat,action_rep,weight = None, reduction = 'none')        
        negativ_pseudo_loss = torch.mean(torch.mul(logpi,advantage))
        self.optimizerPi.zero_grad()
        negativ_pseudo_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm)
        self.optimizerPi.step()
        return negativ_pseudo_loss
    def UpdateV(self,program, target):
        self.optimizerQ.zero_grad()
        loss = F.mse_loss(self.q_function(program), target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_function.parameters(), self.max_norm)
        self.optimizerQ.step()
        return loss
    def UpdateQ(self,program, action,target):
        self.optimizerQ.zero_grad()
        loss = F.mse_loss(self.q_function(program, action).flatten(), target)
        loss.backward()
        self.optimizerQ.step()
        return loss
    def train(self,start,n_episodes):
        listLosspi = []
        listLossQ = []
        self.epsilon_greedy_policy.epsilon = 1
        nb_iterations = []
        for j in range(start,n_episodes+1):
                k = 0
                self.epsilon_greedy_policy.epsilon*=.97
                if self.epsilon_greedy_policy.epsilon<.1:
                    self.epsilon_greedy_policy.epsilon = .1
                #initial program
                program = make_random_source_code(np.random.randint(5))
                signature = execute(compile_program(program))
                #signature = execute(compile_program(program))
                #while self.distance(signature)>2 and k<200:
                dk = self.distance(signature)#distance between state and objective state at iteration k
                dk_list = [dk]
                print("len", len(self.filter(program)))
                while self.end(program) and k<500:
                    action = self.epsilon_greedy_policy([program])
                    transition = super().__call__(action, program)
                    self.buffer.store({"program":program,
                                        "new_program":transition["modified_program"],
                                        "action": action, 
                                        "reward":transition["reward"],
                                        "new_program_signature":transition["modified_program_signature"]})

                    program = transition["modified_program"]
                    signature = transition["modified_program_signature"]
                    dk = self.distance(signature)
                    k+=1


                    samp =self.buffer.sample(min(self.batch_size,len(self.buffer.memory_program)))
                    ap = self.policy(samp["new_program"])
                    targets = samp["reward"] + self.gamma*self.q_function(samp["new_program"],ap).flatten()
                    #print("targets", targets.shape)
                    targets = targets.detach()
                    #Q update 
                    for l in range(self.K):
                        loss = self.UpdateQ(samp["program"],samp["action"], targets)
                    #advantage evaluation
                    #Policy update
                    NegativPseudoLoss = self.UpdatePi(samp) # theta <-- theta + alpha* grad J(theta) using negativpseudoloss for policy pi
                    listLosspi.append(NegativPseudoLoss.item())
                    listLossQ.append(loss.item())
                    dk_list.append(dk)
                    #exit()
                if j%1==0:
                    print(f"episod {j} finished with {k} iterations")
                #if j%1==0:
                    print("episodes", j,f"/{n_episodes}")
                    #print("NegativPseudoLoss",torch.mean(torch.Tensor(listLosspi)))
                    #print("Loss Q", torch.mean(torch.Tensor(listLossQ)))
                    print("epsilon", self.epsilon_greedy_policy.epsilon)

        
                if j%100==0:
                    torch.save(self.policy.state_dict(), os.path.join(self.loadpath,f"pi_load_{j}.pt"))
                    torch.save(self.q_function.state_dict(), os.path.join(self.loadpath,f"v_load_{j}.pt"))

                    torch.save(self.optimizerPi.state_dict(), os.path.join(self.optpath,f"opt_pi_load_{j}.pt"))
                    torch.save(self.optimizerQ.state_dict(), os.path.join(self.optpath,f"opt_q_load_{j}.pt"))
                nb_iterations.append(k)
                if j%5:
                    plt.figure()
                    plt.plot(nb_iterations)
                    plt.title("Number of step per episode")
                    plt.savefig("image/nb_iterations")
                    plt.xlabel("episode")
                    
                    plt.close()

                    plt.figure()
                    plt.plot(listLosspi)
                    plt.title("Pseudo Loss Pi")
                    plt.savefig("image/losspi")
                    plt.xlabel("episode")
                    plt.close()

                    plt.figure()
                    plt.plot(listLossQ)
                    plt.title("Pseudo Loss Q")
                    plt.savefig("image/lossQ")
                    plt.xlabel("episode")
                    plt.close()
                plt.figure()
                plt.semilogy(dk_list)
                plt.title("Weighted distance between the current program signature and the objective signature at time k during the episod")
                plt.savefig(f"image/dk/{j}")
                plt.xlabel("episode k")
                
                plt.close()

                    


        return nb_iterations        





class Buffer:
    def __init__(self, maxsize = 10000):
        self.memory_program = []
        self.memory_action = {"line":[], "change":[]}
        self.memory_newprogram = []
        self.memory_reward = []
        self.maxsize = maxsize
    def store(self,sample):
        #for j in range(len(sample["program"])):
        self.memory_program.append(sample["program"])
        self.memory_action["line"].append(sample["action"]["line"])
        self.memory_action["change"].append(sample["action"]["change"])
        self.memory_newprogram.append(sample["new_program"])
        self.memory_reward.append(sample["reward"])
        self.eviction()
    def eviction(self):
        if len(self.memory_program)>self.maxsize:
            self.memory_program = self.memory_program[-self.maxsize:]
            self.memory_action["line"] = self.memory_action["line"][-self.maxsize:]
            self.memory_action["change"] = self.memory_action["change"][-self.maxsize:]
            self.memory_newprogram = self.memory_newprogram[-self.maxsize:]
            self.memory_reward = self.memory_reward[-self.maxsize:]

    def sample(self,N):
        assert(type(N)==int and N>0 and N<=len(self.memory_program))
        selection = torch.randint(0,len(self.memory_program),(N,))
        #print("selection", selection)
        program = [self.memory_program[j] for j in selection]
        action = {"line":[self.memory_action["line"][j] for j in selection],
                  "change":[self.memory_action["change"][j] for j in selection]}
        newprogram = [self.memory_newprogram[j] for j in selection]
        reward = torch.Tensor([self.memory_reward[j] for j in selection]).to(device)
        #renvoie un tuple de 4 tenseurs (s,a,s',r)
        sample = {"program": program,
                  "action": action,
                  "new_program": newprogram,
                  "reward": reward
                  }
        #return program, action, newprogram, reward
        #print("samp", sample)
        return sample
    def print (self):

        print("self.memory_program    "      ,self.memory_program )
        print("self.memory_action   "      ,self.memory_action )
        print("self.memory_newprogram "      ,self.memory_newprogram)
        print("self.memory_reward   "      ,self.memory_reward )
