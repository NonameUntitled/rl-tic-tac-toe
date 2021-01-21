import numpy as np
from collections import defaultdict


class BasicAgent:
    def __init__(self, player_num):
        self.player_num = player_num

    def chose_action(self, current_state, possible_actions):
        pass

    def learn(self, reward):
        pass


class RandomAgent(BasicAgent):
    def __init__(self, player_num):
        super(RandomAgent, self).__init__(player_num)

    def chose_action(self, current_state, possible_actions):
        action_idx = np.random.randint(0, len(possible_actions))
        return possible_actions[action_idx]


class BasicVFuncAgent(BasicAgent):
    def __init__(self, player_num, learning_rate):
        super(BasicVFuncAgent, self).__init__(player_num)
        self.learning_rate = learning_rate
        self.value_function = defaultdict(lambda: 0.0)

        self.prev_state = []

    def chose_action(self, current_state, possible_actions):
        best_actions = []
        best_states = []

        best_value = -200

        for action in possible_actions:
            tmp = list(current_state)
            tmp[action] = self.player_num
            possible_state = tuple(tmp)

            if self.value_function[possible_state] == best_value:
                best_states.append(possible_state)
                best_actions.append(action)
            elif self.value_function[possible_state] > best_value:
                best_states = [possible_state]
                best_actions = [action]
                best_value = self.value_function[possible_state]

        action_idx = np.random.randint(0, len(best_actions))
        self.prev_state.append(best_states[action_idx])

        return best_actions[action_idx]

    def learn(self, reward):
        for idx, state in enumerate(self.prev_state[::-1]):
            self.value_function[state] = self.value_function[state] + self.learning_rate ** (idx + 1) * (
                        reward - self.value_function[state])


class PlayerAgent(BasicAgent):
    def __init__(self, player_num):
        super(PlayerAgent, self).__init__(player_num)

    def chose_action(self, current_state, possible_actions):
        print('Current state:')
        for line in current_state:
            print(line)
        print(f'Possible actions: {possible_actions}')

        action = -1
        while action not in possible_actions:
            action = int(input('Enter action: '))

        return action


# TODO Make adaptive epsilon
class GreedyVFuncAgent(BasicVFuncAgent):
    def __init__(self, player_num, learning_rate, epsilon):
        super(GreedyVFuncAgent, self).__init__(player_num, learning_rate)
        self.epsilon = epsilon

    def chose_action(self, current_state, possible_actions):
        val = np.random.random()
        if val < self.epsilon:
            action_idx = np.random.randint(0, len(possible_actions))
            self.prev_state.append(None)
            return possible_actions[action_idx]
        else:
            best_actions = []
            best_states = []

            best_value = -200

            for action in possible_actions:
                tmp = list(current_state)
                tmp[action] = self.player_num
                possible_state = tuple(tmp)

                if self.value_function[possible_state] == best_value:
                    best_states.append(possible_state)
                    best_actions.append(action)
                elif self.value_function[possible_state] > best_value:
                    best_states = [possible_state]
                    best_actions = [action]
                    best_value = self.value_function[possible_state]

            action_idx = np.random.randint(0, len(best_actions))
            self.prev_state.append(best_states[action_idx])

            return best_actions[action_idx]

    def learn(self, reward):
        for idx, state in enumerate(self.prev_state[::-1]):
            if state is not None:
                self.value_function[state] = self.value_function[state] + self.learning_rate ** (idx + 1) * (
                        reward - self.value_function[state])