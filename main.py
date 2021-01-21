from tqdm import trange

from agents.agents import *
from environment.Environment import Environment

if __name__ == '__main__':
    fst_agent = GreedyVFuncAgent(1, 0.7, 0.1)
    snd_agent = BasicVFuncAgent(2, 0.7)

    for _ in trange(1000):
        env = Environment()

        turn = 0
        while not env.is_finished(env.field):
            # First agents turn
            current_state = env.get_state_description(env.field)
            possible_actions = env.get_possible_actions(fst_agent.player_num)
            chosen_action = fst_agent.chose_action(current_state, possible_actions)

            prev_field, action, new_field, is_done, rewards = env.step(chosen_action, fst_agent.player_num)
            fst_reward, snd_reward = rewards

            if is_done:
                fst_agent.learn(fst_reward)
                snd_agent.learn(snd_reward)
                break

            # Second agents turn
            current_state = env.get_state_description(env.field)
            possible_actions = env.get_possible_actions(snd_agent.player_num)

            chosen_action = snd_agent.chose_action(current_state, possible_actions)

            prev_field, action, new_field, is_done, rewards = env.step(chosen_action, snd_agent.player_num)
            fst_reward, snd_reward = rewards

            if is_done:
                fst_agent.learn(fst_reward)
                snd_agent.learn(snd_reward)
                break

            turn += 1

    snd_agent = PlayerAgent(2)

    env = Environment()

    turn = 0
    while not env.is_finished(env.field):
        # First agents turn
        current_state = env.get_state_description(env.field)
        possible_actions = env.get_possible_actions(fst_agent.player_num)
        chosen_action = fst_agent.chose_action(current_state, possible_actions)

        prev_field, action, new_field, is_done, rewards = env.step(chosen_action, fst_agent.player_num)
        fst_reward, snd_reward = rewards

        if is_done:
            fst_agent.learn(fst_reward)
            snd_agent.learn(snd_reward)
            break

        # Second agents turn
        current_state = env.field
        possible_actions = env.get_possible_actions(snd_agent.player_num)

        chosen_action = snd_agent.chose_action(current_state, possible_actions)

        prev_field, action, new_field, is_done, rewards = env.step(chosen_action, snd_agent.player_num)
        fst_reward, snd_reward = rewards

        if is_done:
            fst_agent.learn(fst_reward)
            snd_agent.learn(snd_reward)
            break

        turn += 1


    print(fst_agent.value_function.values())
