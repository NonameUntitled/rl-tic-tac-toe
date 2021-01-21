import copy
import numpy as np


class Environment:
    def __init__(self):
        self.field = np.zeros((3, 3))

    def print_env(self):
        for line in self.field:
            print(line)
        print()

    @staticmethod
    def is_finished(field):
        """
        Tells whether game is over or not
        :param field: Tic-tac-toe field
        :return: True if the game is over, False otherwise
        """
        # For Xs
        for i in range(3):
            if np.all(field[i, :] == 1) or np.all(field[:, i] == 1):
                return True
        # For Os
        for i in range(3):
            if np.all(field[i, :] == 2) or np.all(field[:, i] == 2):
                return True
        # For both
        if (field[0, 0] == field[1, 1] == field[2, 2] != 0) \
                or (field[2, 0] == field[1, 1] == field[0, 2] != 0):
            return True

        return np.all(field != 0)

    @staticmethod
    def detect_winner(field):
        """
        Detects game winner
        :param field: Final tic-tac-toe field
        :return: 1 if Xs player is a winner, 2 if Os player is a winner, 0 if it is tie
        """
        # For Xs
        for i in range(3):
            if np.all(field[i, :] == 1) or np.all(field[:, i] == 1):
                return 1
        if (field[0, 0] == field[1, 1] == field[2, 2] == 1) or (field[0, 2] == field[1, 1] == field[2, 0] == 1):
            return 1

        # For Ys
        for i in range(3):
            if np.all(field[i, :] == 2) or np.all(field[:, i] == 2):
                return 2
        if (field[0, 0] == field[1, 1] == field[2, 2] == 2) or (field[0, 2] == field[1, 1] == field[2, 0] == 2):
            return 2

        # For tie
        return 0

    @staticmethod
    def get_reward(field):
        """
        Computes rewards in current state for two players
        :return: tuple of (first_player_reward, second_player_reward)
        """
        if not Environment.is_finished(field):
            return 0, 0
        else:
            if Environment.detect_winner(field) == 1:
                return 10, -10
            elif Environment.detect_winner(field) == 2:
                return -10, 10
            else:
                return -3, -3

    def step(self, action, player_num):
        """
        Changes field according to a player step
        :param action: Place where to put symbol
        :param player_num: Xs or Os (1 or 2)
        :return: prev_state, action, new_state, is_done, rewards
        """
        assert 0 <= action < 9
        assert self.field[action // 3, action % 3] == 0
        assert player_num == 1 or player_num == 2

        old_field = copy.deepcopy(self.field)

        new_field = copy.deepcopy(self.field)
        new_field[action // 3, action % 3] = player_num
        self.field = new_field

        is_done = self.is_finished(new_field)

        player_rewards = self.get_reward(new_field)

        return old_field, action, new_field, is_done, tuple(player_rewards)

    def get_possible_actions(self, symbol_value):
        possible_actions = []

        for i in range(3 * 3):
            if self.field[i // 3, i % 3] == 0:
                possible_actions.append(i)

                possible_next_state = copy.deepcopy(self.field)
                possible_next_state[i // 3, i % 3] = symbol_value

                if self.is_finished(possible_next_state):
                    return [i]

        return possible_actions

    @staticmethod
    def get_state_description(field):
        description = []
        for i in range(3 * 3):
            description.append(field[i // 3, i % 3])
        return tuple(description)


if __name__ == '__main__':
    a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert not Environment.is_finished(a)

    a = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 1

    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 1

    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 1

    a = np.array([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 1

    a = np.array([[2, 0, 1], [0, 2, 0], [1, 0, 2]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 2

    a = np.array([
        [2, 1, 1],
        [1, 2, 2],
        [1, 2, 1]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 0

    a = np.array([
        [1, 2, 1],
        [1, 1, 2],
        [2, 1, 2]])
    assert Environment.is_finished(a)
    assert Environment.detect_winner(a) == 0

    a = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 1]])
    assert not Environment.is_finished(a)
