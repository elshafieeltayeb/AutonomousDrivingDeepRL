import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", default=None)
    parser.add_argument("--model_path", default=None)

    parser.add_argument("--train_dqn", action="store_true", help="whether train DQN")
    parser.add_argument("--test_dqn", action="store_true", help="whether test DQN")
    parser.add_argument(
        "--dqn_type", default=None, help="[DQN, DoubleDQN, DuelDQN, DDDQN]"
    )

    parser.add_argument("--video_dir", default=None, help="output video directory")
    parser.add_argument(
        "--do_render", action="store_true", help="whether render environment"
    )

    parser.add_argument(
        "--with_lane_accuracy", default=False, help="with lane accuracy rewards"
    )

    args = parser.parse_args()
    return args
