from configparser import ConfigParser
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.linalg import norm
from .car_agent import CarAgent
from os.path import dirname, abspath, join
import sys
import airsim
import time
import math

sys.path.append("..")


class AirSimCarEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        hit_rate_file_path: Optional[str] = None,
        variability_rate_file_path: Optional[str] = None,
        reward_with_centered_lane_accuracy=False,
    ):
        super().__init__()
        self.hit_rate_file_path = hit_rate_file_path
        self.variability_rate_file_path = variability_rate_file_path
        self.reward_with_centered_lane_accuracy = reward_with_centered_lane_accuracy
        self.ip_address = ("127.0.0.1",)
        # self.image_shape = (84, 84, 1),
        # Define action and observation space
        # They must be gym.spaces objects

        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), "config.ini"))

        # Using discrete actions
        # TODO Compute number of actions from other settings
        self.action_space = spaces.Discrete(int(config["car_agent"]["actions"]))

        self.image_height = int(config["airsim_settings"]["image_height"])
        self.image_width = int(config["airsim_settings"]["image_width"])
        self.image_channels = int(config["airsim_settings"]["image_channels"])
        image_shape = (self.image_height, self.image_width, self.image_channels)

        self.track_width = float(config["airsim_settings"]["track_width"])

        # Using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )

        self.car_agent = CarAgent()

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()

        reward, done = (
            self._compute_reward_with_centered_lane()
            if self.reward_with_centered_lane_accuracy
            else self._compute_reward()
        )
        return obs, reward, done, done, self.state

    def reset(self, step=None, avg_return=None):
        if step:
            self._add_hit_rate(
                self.state["pose"].position.x_val,
                self.state["pose"].position.y_val,
                step,
            )
            if avg_return:
                self._add_variability_rate(avg_return, step)
        self._setup_car()
        self._do_action(1)
        return self._get_obs(), self.state

    def render(self, mode="human"):
        return  # nothing

    def close(self):
        self.car_agent.reset()
        return

    def _compute_reward(self, car_state=None):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1),
                (130, -1),
                (130, 125),
                (0, 125),
                (0, -1),
                (130, -1),
                (130, -128),
                (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )
        if dist > THRESH_DIST:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed
        done = 0
        if reward < -2:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done

    def _compute_reward_with_centered_lane(self, car_state=None):
        MAX_SPEED = 300
        MIN_SPEED = 10

        target_x_val = 130
        target_y_val = -1

        current_distance_to_target = target_x_val - self.state["pose"].position.x_val
        current_distance_to_lane_center = (
            target_y_val - self.state["pose"].position.y_val
        )

        prev_distance_to_target = target_x_val - self.state["prev_pose"].position.x_val
        # prev_distance_to_lane_center = target_y_val - self.state["prev_pose"].position.y_val

        reward = -3
        if (
            current_distance_to_target < prev_distance_to_target
            and current_distance_to_target < target_x_val
        ):
            reward_discount = 0
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            if abs(current_distance_to_lane_center > target_y_val) > 2:
                reward_discount = -1 * abs(
                    current_distance_to_lane_center > target_y_val
                )
            reward = reward_speed + reward_discount + 1
        elif prev_distance_to_target < current_distance_to_target < target_x_val:
            reward_discount = -1
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            if abs(current_distance_to_lane_center > target_y_val) > 2:
                reward_discount = -1 * abs(
                    current_distance_to_lane_center > target_y_val
                )
            reward = reward_speed + reward_discount + 1

        done = 0
        if reward < -2:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done

    def _isDone(self, car_state, car_controls, reward):
        if reward < 0:
            return True

        car_pos = car_state.kinematics_estimated.position
        car_point = [car_pos.x_val, car_pos.y_val]
        destination = self.car_agent.simGetWayPoints()[-1]
        distance = norm(car_point - destination)
        if distance < 5:  # 5m close to the destination, stop
            return True

        return False

    def _setup_car(self):
        self.car_agent.restart()
        self.car_agent.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car_agent.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car_agent.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car_agent.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car_agent.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car_agent.simGetCollisionInfo().has_collided

        return image

    def _add_hit_rate(self, x_val, y_val, step):
        with open(self.hit_rate_file_path, "a") as f:
            f.write(f"\n{ x_val},{y_val},{step}")

    def _add_variability_rate(self, avg_return, step):
        with open(self.variability_rate_file_path, "a") as f:
            f.write(f"\n{step},{avg_return}")
