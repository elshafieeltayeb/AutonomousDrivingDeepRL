# Autonomous Driving using Deep RL
The repository houses an implementation of Deep Reinforcement Learning (DRL) integrated with a simulation environment for Autonomous Driving Systems. 
This comprehensive project combines the power of DRL algorithms with a sophisticated simulation platform, 
allowing researchers and developers to explore and enhance decision-making processes in autonomous vehicles. 
The codebase includes implementations of DRL algorithms, with a particular focus on Deep Q-Learning, 
tailored for the unique challenges posed by autonomous driving scenarios. 

The 3D environments used in this project including the AirsSim plugin [AirSim](https://github.com/microsoft/AirSim),
are included in the [releases repository](https://github.com/Microsoft/AirSim/releases)


The followings are used to facilitate training models:

*[OpenAI Gym](https://gym.openai.com/)
*[AirSim](https://github.com/microsoft/AirSim)

DRL algorithms:
* DQN

* Download the AirSim environment from [releases repository](https://github.com/Microsoft/AirSim/releases)
* Run the environment Binary 
* run `poetry install`
* run `pip install msgpack-rpc-python` &  `pip install airsim`
* (optional) install tensorflow using `conda install -c apple tensorflow-deps` then `pip install tensorflow-macos`


## How to train
* Config.ini file to includes some settings, like input image resolution, action space of the agent, etc.
```
# settings related to UE4/airsim 
[airsim_settings] 
image_height = 144
image_width = 256
image_channels = 3
waypoint_regex = WayPoint.*
track_width = 12 

# settings related to training car agent
[car_agent]
# what are adjusted in driving the car
# 0: change steering only, 1(not supported yet): change throttle and steering,
# 2(not supported yet): change throttle, steering, and brake
action_mode = 0 
# steering value from left to right in range [-1, 1] 
# e.g: 0.3 means steering is from -0.3 (left) to 0.3 (right)
steering_max = 0.3
# the granularity of steering range as we use discrete values for steering
# e.g: 7 will produce discrete steering(with max=0.3) actions as: -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3
steering_granularity = 7 
# car's acceleration, now it is fixed, but will add a range of values later 
fixed_throttle = 0.75 
# total actions of car agent, update this accordingly when changing the above settings
actions = 7 
```
* Training
  ```
  python main.py --train_dqn --dqn_type=DQN || DoubleDQN || DuelDQN || DDDQN --folder_name=[YOUR NAME]
  python run.py --train_dqn --dqn_type=DQN --folder_name=training_data
  ```
* Testing
  ```
  python main.py --test_dqn --dqn_type=DQN || DoubleDQN || DuelDQN || DDDQN --model_path=[YOUR MODEL PATH]
  python main.py --test_dqn --dqn_type=DDDQN --model_path=./model/best/dqn.cpt
  ```
* Plot
  ```
  python plots/rl_performance_plots.py
  ```
## References
* [create custom open ai gym environment](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
