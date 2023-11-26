"""

### NOTICE ###
You DO NOT need to upload this file

"""

from gym_airsim.airsim_car_env import AirSimCarEnv
from parse_command import parse
from test import test
import warnings
import datetime
import pathlib

current_working_dir = pathlib.Path().resolve()
warnings.filterwarnings("ignore")


def run(args):
    "******Deep Q Learning******"
    if args.train_dqn:
        hit_rate_file_path = (
            str(current_working_dir)
            + f'/report/hit_rate_{datetime.datetime.now().strftime("%m%d%Y%H%M%S")}.csv'
        )
        with open(hit_rate_file_path, "w") as f:
            f.write(f"x_val,y_val,step")
        variability_rate_file_path = (
            str(current_working_dir)
            + f'/report/variability_rate_{datetime.datetime.now().strftime("%m%d%Y%H%M%S")}.csv'
        )
        with open(variability_rate_file_path, "w") as f:
            f.write(f"Metrics/EnvironmentSteps,Metrics/AverageReturn")
        env = AirSimCarEnv(
            hit_rate_file_path=hit_rate_file_path,
            variability_rate_file_path=variability_rate_file_path,
            reward_with_centered_lane_accuracy=args.with_lane_accuracy,
        )
        from rl.agent_dqn import AgentDQN

        agent = AgentDQN(env, args)
        agent.train()
    if args.test_dqn:
        env = AirSimCarEnv()
        from rl.agent_dqn import AgentDQN

        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == "__main__":
    args = parse()
    run(args)
