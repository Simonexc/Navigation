from dqn_agent import Agent, TAU
import torch.nn.functional as F


class DoubleDQNAgent(Agent):
    
    def calculate_Q_values(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # get best action (from local network) in a vertical vector form
        best_next_action = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        # get predicted Q-value for the next state (from target network) based on local network best action
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_next_action)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        return Q_expected, Q_targets
