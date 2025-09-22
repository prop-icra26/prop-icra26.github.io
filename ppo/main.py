import torch 
import matplotlib.pyplot as plt
import tqdm
import gymnasium as gym 
import panda_gym
import os 
from algos.networks import PPOK, PPO
from argparse import ArgumentParser
import numpy as np 
import pickle
import sys 

    
def baseline(args):
    env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human", control_type="joints")
    os.makedirs("./weights/", exist_ok=True)
    state, _ = env.reset()
    state_dim = 0
    for k, v in state.items():
        state_dim += len(v)
    action_dim = env.action_space.shape[0] # type: ignore
    device = torch.device("cpu")
    ppo = PPO(state_dim + args.keysize, action_dim, 1e-3, 1e-3, 0.95, 5, 0.2, True, action_std_init=0.5, device=device)
    key = np.random.randint(0, 2, (args.keysize,)).astype(float)
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    if args.plot:
        plt.ion()
    rolling_rewards = []
    state_errors = []
    rolling_rewards_idx = 0
    all_rewards1 = []
    all_fse1 = []
    all_rewards2 = []
    all_fse2 = []
    all_rewards3 = []
    all_fse3 = []

    for e in tbar:
        # if e % 300 == 10:
        #     env = gym.make("PandaReach-v3", reward_type="sparse")
        # if e > 9000 or e % 300 == 0:
        #     env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human")
        fake_key = np.random.randint(0, 2, (args.keysize,)).astype(float)
        close_key = np.copy(key)
        idx = np.random.randint(0, args.keysize, 1)
        close_key[idx] = 1 - close_key[idx]
        
        # repeat this process below for key, fake_key, and close_key
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dist_before = np.linalg.norm(state["achieved_goal"] - state["desired_goal"])
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"], key))
        while (not finished) and t < T:
            action = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            dist_after = np.linalg.norm(next_state["achieved_goal"] - next_state["desired_goal"])
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"], key))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4 - args.keysize:-1 - args.keysize] - state[-6 - args.keysize:-3 - args.keysize])
        if len(rolling_rewards) == 50:
            rolling_rewards[rolling_rewards_idx] = net_reward
            state_errors[rolling_rewards_idx] = state_error
            rolling_rewards_idx = (rolling_rewards_idx + 1) % 50
        else:
            rolling_rewards.append(net_reward)
            state_errors.append(state_error)
        all_rewards1.append(net_reward)
        all_fse1.append(state_error)
        tbar.set_description(f"{net_reward:2.4f}")
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dgoal = state["desired_goal"]
        dgoal[0:2] *= -1.0
        dist_before = np.linalg.norm(state["achieved_goal"] - dgoal)
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"], fake_key))
        while (not finished) and t < T:
            action = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            dgoal = next_state["desired_goal"]
            dgoal[0:2] *= -1.0
            dist_after = np.linalg.norm(next_state["achieved_goal"] - dgoal)
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"], fake_key))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])
        all_rewards2.append(net_reward)
        all_fse2.append(state_error)
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dgoal = state["desired_goal"]
        dgoal[0:2] *= -1.0
        dist_before = np.linalg.norm(state["achieved_goal"] - dgoal)
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"], close_key))
        while (not finished) and t < T:
            action = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            dgoal = next_state["desired_goal"]
            dgoal[0:2] *= -1.0
            dist_after = np.linalg.norm(next_state["achieved_goal"] - dgoal)
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"], close_key))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])

        all_rewards3.append(net_reward)        
        all_fse3.append(state_error)        
        if args.plot:
            plt.subplot(2, 1, 1)
            plt.plot(e, np.mean(rolling_rewards), 'r.')
            plt.title("net reward")
            plt.xlim([max(0, e - 1000), e])
            plt.subplot(2, 1, 2)
            plt.plot(e, np.mean(state_errors), 'b.')
            plt.title("final state error")
            plt.xlim([max(0, e - 1000), e])
            plt.pause(0.01)
        if t <= 1:
            continue
        ppo.update()
        if e % (args.epochs // 10) == 0 and e != 0:
            ppo.decay_action_std(0.05, 0.01)
        
        if e % (args.epochs // 5) == 0 and e != 0:
            torch.save(ppo.policy.state_dict(), f"./weights/ppo_baseline_PandaReach-v3_{args.filename}_{e}.pt")
    
    os.makedirs("data/ppo", exist_ok=True)
    with open(f"data/ppo/baseline_{args.filename}_{args.keysize}.pkl", "wb") as fh:
        pickle.dump({"key": key, "rewards": [all_rewards1, all_rewards2, all_rewards3], "errors": [all_fse1, all_fse2, all_fse3]}, fh)
    return 
    
def main(args):
    env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human", control_type="joints")
    os.makedirs("./weights/", exist_ok=True)
    state, _ = env.reset()
    state_dim = 0
    for k, v in state.items():
        state_dim += len(v)
    action_dim = env.action_space.shape[0] # type: ignore
    device = torch.device("cpu")
    ppo = PPOK(state_dim, action_dim, args.keysize, 1e-3, 0.95, 5, 0.2, True, action_std_init=0.5, device=device)
    key = torch.randint(0, 2, (args.keysize,), dtype=torch.float)
    tbar = tqdm.trange(args.epochs, file=sys.stdout)
    if args.plot:
        plt.ion()
    rolling_rewards = []
    state_errors = []
    rolling_rewards_idx = 0
    all_rewards1 = []
    all_fse1 = []
    all_rewards2 = []
    all_fse2 = []
    all_rewards3 = []
    all_fse3 = []

    for e in tbar:
        # if e % 300 == 10:
        #     env = gym.make("PandaReach-v3", reward_type="sparse")
        # if e > 9000 or e % 300 == 0:
        #     env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human")
        fake_key = torch.randint(0, 2, (args.keysize,), dtype=torch.float, device=device)
        close_key = torch.clone(key)
        idx = np.random.randint(0, args.keysize, 1)
        close_key[idx] = 1 - close_key[idx]
        
        # repeat this process below for key, fake_key, and close_key
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dist_before = np.linalg.norm(state["achieved_goal"] - state["desired_goal"])
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        while (not finished) and t < T:
            action = ppo.select_action(state, key)
            next_state, reward, done, truncated, _ = env.step(action)
            dist_after = np.linalg.norm(next_state["achieved_goal"] - next_state["desired_goal"])
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])
        if len(rolling_rewards) == 50:
            rolling_rewards[rolling_rewards_idx] = net_reward
            state_errors[rolling_rewards_idx] = state_error
            rolling_rewards_idx = (rolling_rewards_idx + 1) % 50
        else:
            rolling_rewards.append(net_reward)
            state_errors.append(state_error)
        all_rewards1.append(net_reward)
        all_fse1.append(state_error)
        tbar.set_description(f"{net_reward:2.4f}")
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dgoal = state["desired_goal"]
        dgoal[0:2] *= -1.0
        dist_before = np.linalg.norm(state["achieved_goal"] - dgoal)
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        while (not finished) and t < T:
            action = ppo.select_action(state, fake_key)
            next_state, reward, done, truncated, _ = env.step(action)
            dgoal = next_state["desired_goal"]
            dgoal[0:2] *= -1.0
            dist_after = np.linalg.norm(next_state["achieved_goal"] - dgoal)
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])
        all_rewards2.append(net_reward)
        all_fse2.append(state_error)
        
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dgoal = state["desired_goal"]
        dgoal[0:2] *= -1.0
        dist_before = np.linalg.norm(state["achieved_goal"] - dgoal)
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        while (not finished) and t < T:
            action = ppo.select_action(state, close_key)
            next_state, reward, done, truncated, _ = env.step(action)
            dgoal = next_state["desired_goal"]
            dgoal[0:2] *= -1.0
            dist_after = np.linalg.norm(next_state["achieved_goal"] - dgoal)
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])

        all_rewards3.append(net_reward)        
        all_fse3.append(state_error)        
        if args.plot:
            plt.subplot(2, 1, 1)
            plt.plot(e, np.mean(rolling_rewards), 'r.')
            plt.title("net reward")
            plt.xlim([max(0, e - 1000), e])
            plt.subplot(2, 1, 2)
            plt.plot(e, np.mean(state_errors), 'b.')
            plt.title("final state error")
            plt.xlim([max(0, e - 1000), e])
            plt.pause(0.01)
        if t <= 1:
            continue
        ppo.update()
        if e % (args.epochs // 10) == 0 and e != 0:
            ppo.decay_action_std(0.05, 0.01)
        
        if e % (args.epochs // 5) == 0 and e != 0:
            torch.save(ppo.policy.state_dict(), f"./weights/ppo_ours_PandaReach-v3_{args.filename}_{e}.pt")
    
    os.makedirs("data/ppo", exist_ok=True)
    with open(f"data/ppo/ours_{args.filename}_{args.keysize}.pkl", "wb") as fh:
        pickle.dump({"key": key.cpu().detach().numpy(), "rewards": [all_rewards1, all_rewards2, all_rewards3], "errors": [all_fse1, all_fse2, all_fse3]}, fh)
    return 




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--keysize", type=int, default=32)
    parser.add_argument("--no-print", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()
    if args.no_print:
        f = open("/dev/null", "w")
        sys.stdout = f
        baseline(args)
        main(args) # ours
        f.close()
    else:
        baseline(args)
        main(args) # ours