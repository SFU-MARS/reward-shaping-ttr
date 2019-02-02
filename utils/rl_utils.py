import numpy as np

###########
# Rollout #
###########
def rollout(policy, start_state, problem, return_traj=False, return_actions=False, return_rewards=False):
    traj = list()
    actions = list()
    rewards = list()
    state = problem.reset_to_state(start_state)
    traj.append(state)

    if problem.env_name == 'DrivingOrigin-v0':
        while True:
            action, _ = policy.act(stochastic=False,
                                   ob=state)
            state, reward, done, _ = problem.step(action)
            traj.append(state)
            actions.append(action)
            rewards.append(reward)
            if done and reward > 0:
                if return_rewards:
                    return True, np.stack(traj), np.stack(actions), np.stack(rewards)
                if return_actions:
                    return True, np.stack(traj), np.stack(actions)
                if return_traj:
                    return True, np.stack(traj)
                else:
                    return True

            elif done:
                if return_rewards:
                    return False, np.stack(traj), np.stack(actions), np.stack(rewards)
                if return_actions:
                    return False, np.stack(traj), np.stack(actions)
                if return_traj:
                    return False, np.stack(traj)
                else:
                    return False

    elif problem.env_name == 'PlanarQuad-v0':
        rollout_count = 0
        while True:
            action, _ = policy.act(stochastic=False,
                                   ob=state)
            state, reward, done, _ = problem.step(action)
            rollout_count += 1
            traj.append(state)
            actions.append(action)
            rewards.append(reward)
            if done and problem.env.unwrapped._in_goal(state):
                if return_rewards:
                    return True, np.stack(traj), np.stack(actions), np.stack(rewards),rollout_count
                if return_actions:
                    return True, np.stack(traj), np.stack(actions)
                if return_traj:
                    return True, np.stack(traj)
                else:
                    return True

            elif done:
                if return_rewards:
                    return False, np.stack(traj), np.stack(actions), np.stack(rewards),rollout_count
                if return_actions:
                    return False, np.stack(traj), np.stack(actions)
                if return_traj:
                    return False, np.stack(traj)
                else:
                    return False
            '''
            ## NOTE: new code in case of the infinite loop when 'done' is always false
            elif rollout_count >= 500 and not done:
                if return_rewards:
                    return False, np.stack(traj), np.stack(actions), np.stack(rewards)
                if return_actions:
                    return False, np.stack(traj), np.stack(actions)
                if return_traj:
                    return False, np.stack(traj)
                else:
                    return False
            '''
