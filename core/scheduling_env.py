import os
import sys
import glob
import logging
import time
import gym
import numpy as np
from core.simulator import Simulator


class SchedulingEnv(gym.Env):
    """Scheduling environment that uses the gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, trace_dir, job_sched_algo, action_group_size,
                 logging_level, reward_window_length=10, random_runtimes=False,
                 fixed_seed=0, allocation_window=1000, model_assignment='',
                 batching=False, batching_algo=None, profiling_data=None,
                 allowed_variants_path=None, max_batch_size=None, ewma_decay=None,
                 infaas_slack=None, infaas_downscale_slack=None):
        super(SchedulingEnv, self).__init__()

        logging.basicConfig(level=logging.INFO)
        np.set_printoptions(threshold=sys.maxsize)

        logfile_name = os.path.join('logs', 'reward', str(time.time()) + '.txt')
        self.logfile = open(logfile_name, mode='w')
        
        self.n_executors = len(glob.glob(os.path.join(os.getcwd(), trace_dir, '*.txt')))
        self.n_accelerators = 4
        self.n_qos_levels = 1
        self.action_group_size = action_group_size

        self.allocation_window = allocation_window
        
        self.max_no_of_accelerators = 10
        self.max_runtime = 1000

        self.model_assignment = model_assignment

        # max total predictors for each accelerator type (CPU, GPU, VPU, FPGA) respectively
        self.predictors_max = self.max_no_of_accelerators * np.ones(self.n_accelerators * self.n_qos_levels)
        self.total_predictors = self.max_no_of_accelerators * np.ones(self.n_accelerators)
        
        # simulator environment that our agent will interact with
        self.simulator = Simulator(trace_path=trace_dir, mode='debugging',
                                   job_sched_algo=job_sched_algo,
                                   logging_level=logging_level,
                                   max_acc_per_type=self.max_no_of_accelerators,
                                   predictors_max=self.total_predictors,
                                   n_qos_levels=self.n_qos_levels,
                                   random_runtimes=random_runtimes,
                                   fixed_seed=fixed_seed,
                                   batching=batching,
                                   model_assignment=model_assignment,
                                   batching_algo=batching_algo,
                                   profiling_data=profiling_data,
                                   allowed_variants_path=allowed_variants_path,
                                   max_batch_size=max_batch_size,
                                   ewma_decay=ewma_decay,
                                   infaas_slack=infaas_slack,
                                   infaas_downscale_slack=infaas_downscale_slack)

        # number of steps that we play into the future to get reward
        # this is a tunable parameter
        self.K_steps = reward_window_length

        self.features = (self.n_accelerators * 2) * self.n_qos_levels + 2*self.n_qos_levels + 2

        high_1d = np.ones((self.features))
        high_1d[-1] = 10000
        high_1d[-2] = 10000

        predictors_end_idx = self.n_accelerators * self.n_qos_levels
        runtimes_end_idx = self.n_accelerators * self.n_qos_levels * 2

        high_1d[:predictors_end_idx] = self.max_no_of_accelerators * np.ones((self.n_accelerators*self.n_qos_levels))
        high_1d[predictors_end_idx:runtimes_end_idx] = self.max_runtime * np.ones((self.n_accelerators*self.n_qos_levels))
        
        # we convert it into a 2d array by duplicating the same row, n_executors times
        high_2d = np.tile(high_1d, (self.n_executors+1, 1))

        # in the observation space, we have self.n_executors rows for each executor, and
        # 1 extra row at the end to denote the remaining available predictors in the system
        # for that row, the rest of the columns will stay 0 throughout
        self.observation_space = gym.spaces.Box(low=np.zeros((self.n_executors+1, self.features)), high=high_2d,
                                                shape=(self.n_executors+1, self.features), dtype=np.int)

        action_space_max = np.concatenate(([self.n_executors], self.predictors_max))
        self.action_space = gym.spaces.MultiDiscrete(action_space_max)

        # initializing a random state
        self.state = self.reset()
        
        self.populate_runtime_info()

        # initializing total and failed requests to be zero
        self.state[:, -2:] = np.zeros((self.n_executors+1, 2))

        self.actions_taken = 0
        self.clock = 0

    
    def step(self, action):
        logging.debug('')
        logging.debug('--------------------------')
        logging.debug('Applying action:{}'.format(action))
        logging.debug('--------------------------')
        logging.debug('')

        if not(self.model_assignment == 'ilp'):
            applied_assignment = self.simulator.apply_assignment_vector(action)
            self.state[action[0], 0:self.n_accelerators*self.n_qos_levels] = applied_assignment
            logging.debug(self.state)

        available_predictors = self.simulator.get_available_predictors()
        total_requests_arr = self.simulator.get_total_requests_arr()
        failed_requests_arr = self.simulator.get_failed_requests_arr()
        completed_requests = self.simulator.completed_requests
        qos_stats = self.simulator.get_qos_stats()

        self.state[-1, 0:4] = available_predictors
        self.state[:-1, -1] = failed_requests_arr
        self.state[:-1, -2] = total_requests_arr
        qos_start_idx = self.n_accelerators*self.n_qos_levels*2
        qos_end_idx = qos_start_idx + self.n_qos_levels*2
        self.state[:-1, qos_start_idx:qos_end_idx] = qos_stats        

        self.actions_taken += 1
        if self.actions_taken == self.action_group_size:
            self.clock += self.allocation_window
            self.simulator.reset_request_count()
            self.simulator.simulate_until(self.clock)
            self.actions_taken = 0

        observation = self.state

        # if no more requests then we are done
        done = self.simulator.is_done()

        # return observation, reward, done, self.failed_requests
        return observation, 0, done, {}

    
    def populate_runtime_info(self):
        predictors_end_idx = self.n_accelerators * self.n_qos_levels
        runtimes_end_idx = self.n_accelerators * self.n_qos_levels * 2

        for i in range(self.n_executors):
            logging.debug(self.state)
        return

    
    def trigger_infaas_upscaling(self):
        self.simulator.trigger_infaas_upscaling()

    
    def trigger_infaas_v2_upscaling(self):
        self.simulator.trigger_infaas_v2_upscaling()
    

    def trigger_infaas_v3_upscaling(self):
        self.simulator.trigger_infaas_v3_upscaling()


    def trigger_infaas_downscaling(self):
        self.simulator.trigger_infaas_downscaling()
    

    def trigger_infaas_v2_downscaling(self):
        self.simulator.trigger_infaas_v2_downscaling()
    

    def trigger_infaas_v3_downscaling(self):
        self.simulator.trigger_infaas_v3_downscaling()

    
    def reset(self):
        observation = np.zeros((self.n_executors+1, self.features))
        self.simulator.reset()
        self.state = observation
        self.failed_requests = 0
        self.populate_runtime_info()
        print('Resetting environment')
        return observation

    
    def render(self, mode='human'):
        self.simulator.print_assignment()
        return

    
    def close(self):
        self.logfile.close()
        return
