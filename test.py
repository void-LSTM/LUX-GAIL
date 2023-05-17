import argparse
import importlib
from datetime import datetime
import os
import json
import shutil
from env.lux_env import ProxyEnv, Observation
from opponent.lux_ai.rl_agent.rl_agent import agent as opAgent
from agent_ import Agent, STATE_CHANNELS
from model.policy_network import PolicyNetwork
from config import MODEL_DIR
import torch
from model.policy_network import PolicyNetwork
from model.teacher_network import PolicyNetwork_teacher
from config import *
from agent_constants import *
from model.feature import feature_net
class EvalEnv():
    def __init__(self, args):
        self.args = args
        self.agent0 = self.get_agent()
        self.agent1 = opAgent

    def get_agent(self):
        device = torch.device('cpu')
        policy_net = PolicyNetwork(feature_size=FEATURE_SIZE, num_unit_actions=UNIT_ACTIONS,
                                num_citytile_actions=CITYTILE_ACTIONS)
        teacher_net=PolicyNetwork_teacher(in_channels=STATE_CHANNELS, feature_size=FEATURE_SIZE, layers=LAYERS,
                               num_unit_actions=UNIT_ACTIONS, num_citytile_actions=CITYTILE_ACTIONS)
        checkpoint_teacher = torch.load('D:\LuxAI-main\checkpoint.pth')
        teacher_net.load_state_dict(checkpoint_teacher['model'])
        teacher_net.eval()
        state_model = feature_net(in_channels=STATE_CHANNELS, feature_size=FEATURE_SIZE, layers=LAYERS)
        save_model = teacher_net
        model_dict =  state_model.state_dict()   
        state_dict = {k:v for k,v in save_model.state_dict().items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        state_model.load_state_dict(model_dict)
        checkpoint_policy = torch.load('D:\LuxAI-main-ppo\ppo_checkpoint.pth')
        policy_net.load_state_dict(checkpoint_policy['model'])
        policy_net.eval()
        agent_ = Agent(policy_net, state_model,device, research_th=0.05, research_turn=15)
        
        return agent_

    def run(self):
        # store the replay.json
        replay_path =datetime.now().strftime("%m%d-%H%M")
        replay_path = os.path.join("replays", replay_path)
        agent0_win_path = os.path.join(replay_path, 'agent0')
        agent1_win_path = os.path.join(replay_path, 'agent1')
        if not os.path.exists(replay_path):
            os.makedirs(replay_path)
            os.makedirs(agent0_win_path)
            os.makedirs(agent1_win_path)

        for i in range(self.args.num_games):
            print('Game {}'.format(i))

            env = ProxyEnv(map_size=self.args.map_size)
            conf = env.conf
            obs, _, done = env.reset()
            step = 0
            while not done:
                observation = Observation()
                observation["updates"] = obs[0]["updates"]
                observation["remainingOverageTime"] = 60.
                observation["step"] = step
                actions = [[],[]]
                observation.player = 0
                actions[0],_,_ = self.agent0(observation,conf)
                observation.player = 1
                actions[1] = self.agent1(observation,conf)
                obs, _, reward,done = env.step(actions)
                step += 1
            env.save(replay_path+'/replay_{}.json'.format(i))
            with open(replay_path+'/replay_{}.json'.format(i)) as f:
                game_result = json.load(f)
                game_rewards = game_result["rewards"]
                if game_rewards[0] > game_rewards[1]:
                    print("    Agent0 win")
                elif game_rewards[0] < game_rewards[1]:
                    print("    Agent1 win")
                else:
                    print("    Draw")

        # organize the game result
        total_num = 0
        agent0_win_num = 0
        agent1_win_num = 0
        agent0_win_list = []
        agent1_win_list = []
        folder_dir = replay_path
        for file_dir in os.listdir(folder_dir):
            file_dir = os.path.join(folder_dir, file_dir)
            if file_dir.endswith("json"):
                total_num += 1
                with open(file_dir, "r") as f:
                    game_result = json.load(f)
                game_rewards = game_result["rewards"]
                if game_rewards[0] > game_rewards[1]:
                    agent_win = 0
                else:
                    agent_win = 1
                if agent_win == 0:
                    agent0_win_num += 1
                    agent0_win_list.append(file_dir.split("/")[-1])
                    shutil.move(file_dir, agent0_win_path)
                else:
                    agent1_win_num += 1
                    agent1_win_list.append(file_dir.split("/")[-1])
                    shutil.move(file_dir, agent1_win_path)
        print()
        print("----------- RESULTS -----------")
        print("agent0 - {}".format('agent0'))
        print("agent1 - {}".format('agent1'))
        print()
        print("agent 0 win {} games over {} games".format(agent0_win_num, total_num))
        print("agent 1 win {} games over {} games".format(agent1_win_num, total_num))

        return None
parser = argparse.ArgumentParser(description='Lux AI Evaluation')
parser.add_argument('--num_games', '-n', help='the number of games', default=1, type=int)
parser.add_argument('--map_size', '-s', help='the size of map', default=12,type=int)
parser.add_argument('--seed', '-r', help='the random seed', default=42, type=int)
args = parser.parse_args()                
        
if __name__ == '__main__':
    print(args)
    eval_env = EvalEnv(args)
    eval_env.run()
    print("Done")




