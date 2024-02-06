import numpy as np
import copy

c = 2

class MCTS_ValueOnly:

    class Node:
        def __init__(self, parent, env, action, done):


            self.n = 0
            self.w = 0
            self.q = 0

            self.parent = parent
            self.childrens = []

            self.env = env

            self.action = action

            self.done = done

    
        def UCT(self):
                
            if self.n == 0:
                return np.inf 

            # Upper Confidence Bound 1
            UCB1 = c * np.sqrt( np.log(self.parent.n) / self.n )
        
            # Upper Confidence Trees = MCTS + UCB1
            return self.q + UCB1
    
     
    def __init__(self, env):
        self.N = 0
        
        env = copy.deepcopy(env)
        self.env = env
        self.root = self.Node(parent=None, env=env, action=-1, done=False)

        self.NUM_ROLLOUT_STEPS = 10
        self.gamma = 0.9
        self.discount_factors = np.array([self.gamma**i for i in range(self.NUM_ROLLOUT_STEPS)])

    def run(self, num_iterations: int):
        
        for i in range(num_iterations):
            
            current = self.root

            #
            # Tree Traversal
            #

            while len(current.childrens) != 0:
     
                UCT_values = [node.UCT() for node in current.childrens]

                max_idx = np.argmax(UCT_values)
                current = current.childrens[max_idx]

            #
            # Rollout?
            #   
       
            if current.n == 0:
                
                v = self.rollout(current)

            else:
                
                if not current.done:
                 
                    #
                    # Node expansion
                    #

                    for action in range(self.env.action_space.n):
                        
                        env_current = copy.deepcopy(current.env)
                    
                        next_state, reward, done, _, _ = env_current.step(action)

                        # ignore reward

                        node = self.Node(parent=current, env=env_current, action=action, done=done)
                        current.childrens.append(node)

                    current = current.childrens[0]

                v = self.rollout(current)

            self.backpropagate(current, v)

     
        return self.get_best_action()
        
    
    def rollout(self, node):
        
        if node.done:
            return 0

        env = copy.deepcopy(node.env)

        reward_list = []

        for i in range(self.NUM_ROLLOUT_STEPS):
                      
            action = node.env.action_space.sample()

            next_state, reward, done , _, _ = env.step(action)

            reward_list.append(reward)

            if done:
                break
      
 
        rewards = np.array(reward_list)
        
        # cumulative discounted reward
        G = np.dot(self.discount_factors[:len(reward_list)], rewards)

        return G   
    

    def backpropagate(self, current, v):
        
        while current:
            
            current.n += 1
            
            current.w = current.w + v
            current.q = current.w / current.n

            current = current.parent

         
                
      
    def get_best_action(self):

        #
        # Determine best action
        #

        q_values = [node.q for node in self.root.childrens if node.n != 0]

        # no action can be selected 
        if len(q_values) == 0:
            return 0

        max_idx = np.argmax(q_values)
     
        best_action = self.root.childrens[max_idx].action

        return best_action

        


    

           