# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Bandit Class
class Bandit:
    def __init__(self):
        self.k = 10  #Number of actions/arms
        self.problems = 2000  #Number of probelms 
        self.qstar = np.random.normal(0,1,(self.problems,self.k))  #Assigning actual action values to 10 actions of 2000 problems stored in a 2D Array


    def reward(self,problem, action): #Function to return the reward given problem number and action taken
        return np.random.normal(self.qstar[problem,action],1)   #Return reward -> A random value from a normal distribution with mean = actual action value and variance = 1
    
    def best_action(self,Q): # Function to return the best unbiased action with highest estimated action value
        arr = [] # Store indices (action number) of the actions with highest estimated action values
        for i in range(len(Q)):
            if Q[i] == Q.max():
                arr.append(i)
        return arr[np.random.randint(len(arr))] # Return a random action among actions with highest action value
    
    def ucb(self,Q,NR,t): # Function to return action based on Upper Confidence Bound Policy
        c = 4 #Confidence level
        if NR.min()==0:
            # Return a random action which is taken 0 times
            arr = []
            for j in range(len(Q)):
                if Q[j] == 0:
                    arr.append(j)
            return arr[np.random.randint(len(arr))]
 
        else:
            M = Q + c * np.sqrt(np.divide(np.log(t),NR)) # Calculating uncertainities of all actions and returning the most uncertain action
            return np.argmax(M) 

    
    def bandit(self, epsilon, steps, initial_q, alpha = 0,action_sel = 'og'):  # Bandit function taking epsilon, steps, initial state values, alpha and type of action selection policy as input
        rewards = np.zeros(steps) # Initialise rewards array                   # 'og' == epsilon greedy action selection, 'ucb' == upper confidence bound action selection
        actions = np.zeros(steps) #Intitialise best actions array

        for i in range(self.problems):      # Iterate through all 2000 problems 
            Q = np.ones(self.k)*initial_q   # initial values 
            NR = np.zeros(self.k)           # number of rewards given
            Best_Action = np.argmax(self.qstar[i])  

            for t in range(steps):  #Take 1000 steps for each problem 
                action = None
                if np.random.rand() < epsilon:  #Epsilon-greedy action selection 
                    action = np.random.randint(self.k)
                else:
                    if action_sel=='og':  # Choose greedy action 
                        action = self.best_action(Q)
                    else:
                        action = self.ucb(Q,NR,t) #Choose action according to UCB policy 

                reward = self.reward(i,action)  #Give reward based on action taken 
                NR[action]+=1 #Increement count for that particular action 
                if alpha > 0:
                    Q[action] = Q[action] + (reward - Q[action]) * alpha  #Update Rule for alpha != 0
                else:
                    Q[action] = Q[action] + (reward - Q[action]) / NR[action] #Update rule for alpha=0 (sample average update rule)
                
                rewards[t] += reward # Add cumulative reward in that particular time step
                if action == Best_Action: #Check if action taken was the optimal action
                    actions[t]+=1          #Increement the number of optimal actions for that particular time step 
        
        return np.divide(rewards,self.problems), np.divide(actions,self.problems) # Return average reward/time step and average number of optimal actions/time step

    def runmain(self): # Main function for the Bandit Problem
        avg_reward, percent_best = self.bandit(epsilon=0,steps=1000,initial_q=0)
        avg_reward1, percent_best1 = self.bandit(epsilon=0.01,steps=1000,initial_q=0)
        avg_reward2, percent_best2 = self.bandit(epsilon=0.1,steps=1000,initial_q=0)

        # Plotting graphs 

        #Epsilon-Greedy Action Selection
        plt.figure(figsize=(12,6))
        plt.title("Average Reward")
        plt.plot(avg_reward, 'g', label='epsilon = 0')
        plt.plot(avg_reward1, 'r', label='epsilon = 0.01')
        plt.plot(avg_reward2, 'b', label='epsilon = 0.1')
        plt.ylabel('Average Reward')
        plt.xlabel('Time Steps')
        plt.legend() 
        #plt.show()

        plt.figure(figsize=(12,6))
        plt.title("Percentage of times optimal action selected")
        plt.plot(percent_best*100, 'g', label='epsilon = 0')
        plt.plot(percent_best1*100, 'r', label='epsilon = 0.01')
        plt.plot(percent_best2*100, 'b', label='epsilon = 0.1')
        plt.ylabel('Percentage of Optimal Action')
        plt.xlabel('Time Steps')
        plt.legend() 


        #Optimistic Initial Value Estimation 
        avg_reward3, percent_best3 = self.bandit(epsilon=0.01,steps=1000,initial_q=5,alpha=0.1)
        plt.figure(figsize=(12,6))
        plt.title("Optimistic Initial Value Optimal Action")
        plt.plot(percent_best1*100, 'r', label='Realistic')
        plt.plot(percent_best3*100, 'b', label='Optimistic')
        plt.ylabel('Percentage of Optimal Action')
        plt.xlabel('Time Steps')
        plt.legend() 

        plt.figure(figsize=(12,6))
        plt.title("Optimistic Initial Value Average Reward")
        plt.plot(avg_reward3, 'r', label='UCB for epsilon = 0.01')
        plt.plot(avg_reward1, 'b', label='Greedy action selection for epsilon = 0.01')
        plt.xlabel('Time Steps')
        plt.ylabel('Average Reward')
        plt.legend() 


        #Upper Confidence Bound Action Selection
        avg_reward_ucb, percent_best_ucb = self.bandit(epsilon=0.01,steps=1000,initial_q=0,action_sel='ucb')
        plt.figure(figsize=(12,6))
        plt.title("Upper Confidence Bound Action Selection Optimal Action ")
        plt.plot(percent_best1*100, 'r', label='Greedy Action Selection for epsilon = 0.01')
        plt.plot(percent_best_ucb*100, 'b', label='UCB for epsilon = 0.01')
        plt.ylabel('Percentage of Optimal Action')
        plt.xlabel('Time Steps')
        plt.legend() 

        plt.figure(figsize=(12,6))
        plt.title("Upper Confidence Bound Action Selection Average Reward")
        plt.plot(avg_reward1, 'r', label='Greedy Action Selection for epsilon = 0.01')
        plt.plot(avg_reward_ucb, 'b', label='UCB for epsilon = 0.01')
        plt.ylabel('Average Reward')
        plt.xlabel('Time Steps')
        plt.legend() 

        #plt.show()

    
class MRP:

    def __init__(self): #Initialising instance variables 
        self.actual_values = [1/6, 2/6, 3/6, 4/6, 5/6] # Actual values of states from left to right 
        self.states = ['A', 'B', 'C', 'D', 'E'] #List of states 
        self.toss = [0,1] # 2 possible actions : 0 --> move left, 1--> move right 
        self.values0 = [0.5,0.5,0.5,0.5,0.5] # Initial values for all states 

    def root_mean_sq(self,Q): #Returning RMS Error given predicted state values
        sum = 0
        for i in range(5):
            sum += (np.square(self.actual_values[i]- Q[i]))
        sum = sum/5.0
        return np.sqrt(sum)

    def reward(self,PS,A): #Returning Reward given Present State and action taken 
        # Input as PS and action A
        if PS=='E' and A==1: # Reward = 1 only if Present State = 'E' and we go right 
            return 1,'T'
        elif PS == 'A' and A==0: #Going left from state 'A'
            return 0,'T'
        else: #In all other Present State Action pairs 
            if A == 0: # Moving Left
                return 0,self.states[self.states.index(PS)-1]
            elif A == 1: #Moving Right 
                return 0,self.states[self.states.index(PS)+1]

    def temp_diff(self, alpha = 0, num_epi = 100, s_ini = 0.5,gamma = 1): #TD Function taking intial values, number of episodes, alpha, and gamma as input parameters
        V_predicted = np.ones(5)*s_ini # Initialising predicted state values 
    
        for i in range(num_epi): #Iterate through 100 episodes 
            
            S = 'C' #Initital state = 'C'
            NS = None #Next State intitialised to None
            flag = False #Flag variable to check if episode terminated
            while True: # Each episode
                #Take action 
                A = np.random.randint(2)
                #Observe Reward, Next State 
                R, NS = self.reward(S,A)
                #Observe NS
                if NS == 'T': #Check if Next State a Terminal State 
                    flag = True
                #Update values
                if NS!= 'T':
                    V_predicted[self.states.index(S)] = V_predicted[self.states.index(S)] + alpha*(R + gamma*V_predicted[self.states.index(NS)] - V_predicted[self.states.index(S)])
                else:
                    V_predicted[self.states.index(S)] = V_predicted[self.states.index(S)] + alpha*(R - V_predicted[self.states.index(S)] )

                S = NS #New Present State becomes the Next State obtained

                if flag==True: #If Next State a Terminal State, terminate the episode 
                    break
        
        return V_predicted, self.root_mean_sq(V_predicted) #Return predicted State Values and corresponding RMS Error

    
    def runmain(self): #Main function for TD problem

        pred11, rmse11 = self.temp_diff(alpha=0.1,num_epi=1)
        pred21, rmse21 = self.temp_diff(alpha=0.1,num_epi=10)
        pred31, rmse31 = self.temp_diff(alpha=0.1,num_epi=100)

        #Plotting 
        plt.figure(figsize=(12,6))
        plt.title("Estimated Value")
        plt.plot(self.values0, 'r', label='0')
        plt.plot(pred11, 'b', label='1')
        plt.plot(pred21, 'g', label='10')
        plt.plot(pred31, 'c', label='100')
        plt.plot(self.actual_values, 'k', label='True')
        plt.xlabel('State')
        plt.ylabel('Temporal Difference Estimate / True Values')
        plt.xticks([0,1,2,3,4],['A','B','C','D','E'])
        plt.legend() 


        #Calculating RMS Error for each step and storing it in an array 
        # Doing this process for 100 steps for alpha = 0.15,0.1,0.05
        arr1 = []
        for i in range(100):
            pred,rmse = self.temp_diff(alpha=0.15,num_epi=i)
            arr1.append(rmse)
        arr2 = []
        for i in range(100):
            pred,rmse = self.temp_diff(alpha=0.1,num_epi=i)
            arr2.append(rmse)
        arr3 = []
        for i in range(100):
            pred,rmse = self.temp_diff(alpha=0.05,num_epi=i)
            arr3.append(rmse)
        
        #Plotting RMS Error 
        plt.figure(figsize=(12,6))
        plt.title("Empirical RMS Error Averaged over States")
        plt.plot(arr1, 'b', label='alpha = 0.15')
        plt.plot(arr2, 'g', label='alpha = 0.1')
        plt.plot(arr3, 'c', label='alpha = 0.05')
        plt.xlabel('Number of Episodes')
        plt.ylabel('RMS Error')
        plt.legend()
        #plt.show()


def main():
    b = Bandit()
    b.runmain()

    m = MRP()
    m.runmain()
    
    plt.show()

if __name__=='__main__':
    main()