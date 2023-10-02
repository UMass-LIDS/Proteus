import os
import copy
import glob
import json
import time
import sys
import logging
import argparse
import numpy as np
import core.utils as utils
from core.exceptions import ConfigException
from core.scheduling_env import SchedulingEnv
from core.utils import dict_subtraction
from algorithms.clipper import Clipper
from algorithms.ilp import Ilp


os.environ["GRB_LICENSE_FILE"] = "gurobi/gurobi.lic"

def getargs():
    parser = argparse.ArgumentParser(description='Test scheduler on the simulated BLIS environment.')
    parser.add_argument('--random_runtimes', '-r', required=False,
                        dest='random_runtimes', action='store_true',
                        help='Initializes random runtimes if used. Otherwise, uses static runtimes.')
    parser.add_argument('--action_size', '-a', required=False, default=1,
                        dest='action_size', help='Number of scheduling changes to make in each iteration. ' +
                        'Default value is 15')
    parser.add_argument('--reward_window_length', '-l', required=False, default=10,
                        dest='reward_window_length', help='The number of steps to look out into the future to ' +
                        'calculate the reward of an action. Default value is 10')
    parser.add_argument('--allocation_window', '-w', required=False, default=1000,
                        dest='allocation_window', help='Milliseconds to wait before recalculating allocation. Default is 1000')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--logging_level', required=False, default=logging.INFO)

    parser.set_defaults(random_runtimes=False, batching=False)

    return parser.parse_args()


def validate_config(config: dict, filename: str):
    model_allocation_algos = ['ilp', 'infaas', 'clipper', 'ilp', 'infaas_v2',
                              'sommelier']
    job_sched_algos = ['random', 'round_robin', 'eft_fifo', 'lft_fifo', 'infaas',
                       'canary_routing']
    batching_algorithms = ['disabled', 'accscale', 'aimd', 'infaas', 'nexus']

    if 'profiling_data' not in config:
        raise ConfigException(f'profiling_data not specified in config file: {filename}')

    if 'trace_path' not in config:
        raise ConfigException(f'trace_path not specified in config file: {filename}')

    if 'model_allocation' not in config:
        raise ConfigException(f'model_allocation algorithm not specified in config '
                              f'file: {filename}')

    if 'job_scheduling' not in config:
        raise ConfigException(f'job_scheduling algorithm (query assignment) not specified '
                              f'in config file: {filename}')
    
    if 'batching' not in config:
        raise ConfigException(f'batching algorithm not specified in config file: {filename}'
                              f'\nPossible choices: {batching_algorithms}')

    if 'allowed_variants' not in config:
        raise ConfigException(f'allowed_variants not specified in config file: {filename}')

    if not(os.path.exists(config['allowed_variants'])):
        raise ConfigException(f'allowed_variants path not found: {config["allowed_variants"]}')

    model_allocation = config['model_allocation']
    if model_allocation not in model_allocation_algos:
        raise ConfigException(f'invalid model_allocation algorithm specified: {model_allocation}. '
                              f'\nPossible choices: {model_allocation_algos}')

    if model_allocation in ['ilp', 'sommelier'] and 'beta' not in config:
        raise ConfigException(f'beta value not specified for model allocation  in '
                              f'config file: {filename}\nbeta value is needed if '
                              f'model allocation is one of the following: ilp, sommelier')
    
    if model_allocation in ['ilp', 'sommelier', 'infaas_v2'] and 'solve_interval' not in config:
        raise ConfigException(f'solve_interval not specified for model allocation in '
                              f'config file: {filename}\nsolve_interval value is needed if '
                              f'model allocation is one of the following: ilp, sommelier')
    
    if not(model_allocation in ['ilp', 'sommelier', 'infaas_v2']) and 'solve_interval' in config:
        raise ConfigException(f'unexpected parameter solve_interval specificed in config: {filename}'
                              f'\solve_interval is only needed for ILP or Sommelier')

    if not(model_allocation in ['ilp', 'sommelier']) and 'beta' in config:
        raise ConfigException(f'unexpected parameter beta specificed in config: {filename}'
                              f'\nbeta is only needed for ILP or Sommelier')

    if (model_allocation == 'clipper' or model_allocation == 'sommelier') and 'static_allocation' not in config:
        raise ConfigException(f'expected static_allocation file path for model allocation '
                              f'algorithm: {model_allocation}, in config: {filename}')

    if 'beta' in config and 1 > float(config['beta']) < 0:
        raise ConfigException(f'invalid value for beta parameter: {config["beta"]}'
                              f'Expected a value between 0 and 1')

    job_scheduling = config['job_scheduling']
    if job_scheduling not in job_sched_algos:
        raise ConfigException(f'invalid job_scheduling algorithm specified: {job_scheduling}. '
                              f'\nPossible choices: {job_sched_algos}')

    batching = config['batching']
    if batching not in batching_algorithms:
        raise ConfigException(f'invalid batching algorithm specified: {batching}. '
                              f'\nPossible choices: {batching_algorithms}')

    if 'max_batch_size' in config:
        max_batch_size = config['max_batch_size']
        if max_batch_size <= 0 or not(isinstance(max_batch_size, int)):
            raise ConfigException(f'unexpected max_batch_size: {max_batch_size}, '
                                  f'positive integer expected for max_batch_size')
        
    if 'ewma_decay' in config:
        if config['ewma_decay'] < 0:
            raise ConfigException(f'unexpected ewma_decay: {config["ewma_decay"]}, '
                                  f'non-negative float expected for ewma_decay')
        
    if 'infaas_slack' in config:
        if config['infaas_slack'] <= 0:
            raise ConfigException(f'unexpected infaas_slack: {config["infaas_slack"]}, '
                                  f'positive float expected for infaas_slack')
        
    if 'infaas_downscale_slack' in config:
        if config['infaas_downscale_slack'] <= 0:
            raise ConfigException(f'unexpected infaas_downscale_slack: {config["infaas_downscale_slack"]}, '
                                  f'positive float expected for infaas_downscale_slack')
    
    if 'fixed_seed' in config:
        if 'seed' not in config:
            raise ConfigException(f'fixed_seed is set to true but seed value is not '
                                  f'specified in config: {filename}')
    else:
        if 'seed' in config:
            raise ConfigException(f'unexpected parameter seed specified in config: '
                                  f'{filename} Seed is only expected when fixed_seed '
                                  f'is true in config')


def main(args):
    action_group_size = int(args.action_size)
    reward_window_length = args.reward_window_length
    allocation_window = int(args.allocation_window)
    logging_level = args.logging_level

    with open(args.config_file) as cf:
        config = json.load(cf)

    validate_config(config=config, filename=args.config_file)

    testing_steps = 50 if 'short_run' in config and config['short_run'] == True else 10000
    solve_interval = config['solve_interval'] if 'solve_interval' in config else 0
    model_assignment = config['model_allocation']
    job_scheduling = config['job_scheduling']
    batching_algo = config['batching']
    file_log_level = logging.INFO if 'log_to_file' in config and config['log_to_file'] is True else logging.WARN

    trace_path = config['trace_path']
    rtypes = len(glob.glob(os.path.join(trace_path, '*.txt')))
    fixed_seed = int(config['seed']) if 'seed' in config else 0
    beta = float(config['beta']) if 'beta' in config else -1
    ewma_decay = float(config['ewma_decay']) if 'ewma_decay' in config else None
    infaas_slack = float(config['infaas_slack']) if 'infaas_slack' in config else None
    infaas_downscale_slack = float(config['infaas_downscale_slack']) if 'infaas_downscale_slack' in config else None
    enable_batching = False if batching_algo == 'disabled' else True
    profiling_data = config['profiling_data']
    allowed_variants_path = config['allowed_variants']
    max_batch_size = config['max_batch_size'] if 'max_batch_size' in config else None

    env = SchedulingEnv(trace_dir=trace_path, job_sched_algo=job_scheduling,
                        action_group_size=action_group_size, logging_level=logging_level,
                        reward_window_length=reward_window_length,
                        random_runtimes=args.random_runtimes, fixed_seed=fixed_seed,
                        allocation_window=allocation_window, model_assignment=model_assignment,
                        batching=enable_batching, batching_algo=batching_algo,
                        profiling_data=profiling_data, allowed_variants_path=allowed_variants_path,
                        max_batch_size=max_batch_size, ewma_decay=ewma_decay,
                        infaas_slack=infaas_slack, infaas_downscale_slack=infaas_downscale_slack)

    print()
    print('--------------------------')
    if model_assignment == 'ilp':
        if 'static_allocation' in config:
            ilp = Ilp(allocation_window=allocation_window, beta=beta,
                    starting_allocation=config['static_allocation'],
                    logging_level=logging_level)
        else:
            ilp = Ilp(allocation_window=allocation_window, beta=beta,
                    logging_level=logging_level)
        ilp_applied = True
        print(f'Testing with solution given by ILP (beta={beta})')
    elif model_assignment == 'infaas':
        print('Testing with INFaaS model assignment policy')
    elif model_assignment == 'infaas_v2':
        print('Testing with INFaaS model assignment policy (batching, cost=accuracy drop)')
    elif model_assignment == 'clipper':
        clipper = Clipper(simulator=env.simulator, solution_file=config['static_allocation'])
        print('Testing with Clipper model assignment policy')
    elif model_assignment == 'sommelier':
        ilp = Ilp(allocation_window=allocation_window, beta=beta, logging_level=logging_level,
                  starting_allocation=config['static_allocation'],
                  static='spec_acc')
        ilp_applied = True
        print('Testing with Sommelier model switching policy (spec_acc)')
    else:
        print('Undefined mode, exiting')
        sys.exit(0)
    print('--------------------------')
    print()
    print('Starting in 5 seconds...')
    time.sleep(5)

    logfile_name = 'log_' + model_assignment + '.txt'
    logfile = open('logs/' + logfile_name, mode='w')

    rate_logger = logging.getLogger('Rate logger')
    rate_loggerfile = os.path.join('logs', 'throughput', model_assignment + '.csv')
    rate_logger.addHandler(logging.FileHandler(rate_loggerfile, mode='w'))
    rate_logger.setLevel(file_log_level)
    rate_logger.info(f'wallclock_time,simulation_time,demand,throughput,capacity,'
                     f'effective_accuracy,total_accuracy,successful,dropped,late,'
                     f'estimated_throughput,estimated_effective_accuracy,ilp_utilization,'
                     f'proportional_lb,ilp_demand,demand_ewma')

    rate_logger_per_model = logging.getLogger('Rate logger per model')
    rate_logger_per_model_file = os.path.join('logs', 'throughput_per_model', str(time.time()) + '_' + model_assignment + '.csv')
    rate_logger_per_model.addHandler(
        logging.FileHandler(rate_logger_per_model_file, mode='w'))
    rate_logger_per_model.setLevel(file_log_level)
    rate_logger_per_model.info(f'wallclock_time,simulation_time,model,demand,throughput,'
                               f'capacity,effective_accuracy,total_accuracy,successful,'
                               f'dropped,late')

    observation = env.reset()
    failed_requests = 0
    total_requests = 0
    successful_requests = 0
    total_successful = 0
    total_accuracy = 0
    total_dropped = 0
    total_late = 0
    total_successful_per_model = {}
    total_accuracy_per_model = {}
    total_dropped_per_model = {}
    total_late_per_model = {}
    ilp_rounds = 0
    start = time.time()
    step_time = time.time()
    last_10_demands = []
    for i in range(testing_steps):
        if model_assignment == 'ilp' or model_assignment == 'sommelier':
            if ilp.is_simulator_set() is False:
                ilp.set_simulator(env.simulator)

            if ilp_applied == True:
                period_tuning = solve_interval
                demand_sum = np.sum(observation[:, -2])
                demand = np.array(observation[:rtypes, -2])
                # Convert horizontal array to vertical
                demand = demand[:, None]
                env.simulator.add_moving_demand(demand)
                last_10_demands.append(demand_sum)
                if len(last_10_demands) > 10:
                    last_10_demands.pop(0)
                if 'zipf_exponential_bursty' in trace_path and any(i > 200 for i in last_10_demands):
                    period_tuning = 2
                    if model_assignment == 'sommelier':
                        period_tuning = 2
                elif 'flat' not in trace_path and i >= 60 and i <= 100:
                    period_tuning = solve_interval / 2

                if i % period_tuning == 0 or i == 5 or i == 20:
                    actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators)
                    ilp_rounds += 1
                elif (i == 47 or i == 59 or i == 71 or i == 83) and 'normal_load' in trace_path:
                    actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators)
                    ilp_rounds += 1
                else:
                    actions = None

            if actions is None:
                action = env.simulator.null_action(env.action_space.sample(), 1)
            elif np.sum(actions) == 0:
                # Apply null action
                action = env.action_space.sample()
                for j in range(len(action)):
                    action[j] = 1
            else:
                ilp_applied = False
                # apply actions iteratively
                action = env.action_space.sample()
                isi = i % env.n_executors
                action[0] = isi
                action[1:5] = actions[isi]
                print('action:', action)

                if isi == env.n_executors - 1:
                    ilp_applied = True

        elif 'infaas' in model_assignment:
            print()
            if i % solve_interval == 0:
                if model_assignment == 'infaas':
                    env.trigger_infaas_upscaling()
                    env.trigger_infaas_downscaling()
                elif model_assignment == 'infaas_v2':
                    env.trigger_infaas_v2_upscaling()
                    env.trigger_infaas_v2_downscaling()
                elif model_assignment == 'infaas_v3':
                    env.trigger_infaas_v3_upscaling()
                    env.trigger_infaas_v3_downscaling()
                else:
                    print(f'Invalid verison of INFaaS: {model_assignment}')
            
            num_isi = observation.shape[0] - 1
            # Initially, we give all executors one CPU predictor to start with
            if i < num_isi:
                action = env.action_space.sample()
                action[0] = i
                action[1:5] = [1,0,0,0]
            else:
                # apply null action, as autoscaling is done by now
                action = env.action_space.sample()
                action[0] = 0
                action[1:5] = observation[0, 0:4]
            action = env.simulator.null_action(env.action_space.sample(), 1)
        elif model_assignment == 'clipper':
            clipper.apply_solution()
            action = env.simulator.null_action(env.action_space.sample(), 1)
        else:
            print(f'Unknown model assignment algorithm: {model_assignment}')
            print('Exiting..')
            exit()

        if i % 10 == 0:
            print(f'Testing step: {i} of {testing_steps}')
            print(f'Simulator event queue length: {len(env.simulator.event_queue)}')
            print(f'Time since last print: {(time.time() - step_time):.3f} seconds')
            step_time = time.time()
            print(f'Action taken: {action}')
            # print('State: {}'.format(env.state))
        observation, reward, done, info = env.step(action)
        print(f'available predictors: {env.simulator.available_predictors}')
        available_predictors = env.simulator.available_predictors
        if max(available_predictors) > env.max_no_of_accelerators or min(available_predictors) < 0:
            print(f'Invalid available predictors state: {available_predictors}')
            time.sleep(10)
        env.simulator.print_allocation()
        # read failed and total requests once every 5 actions
        if (i-1) % action_group_size == 0 or done:
            failed_requests += np.sum(observation[:,-1])
            total_requests += np.sum(observation[:,-2])
            successful_requests += env.simulator.get_successful_requests()
            logfile.write(str(total_requests) + ',' +
                          str(failed_requests) + '\n')
            requests_per_model, failed_per_model, accuracy_per_model, successful_per_model = env.simulator.get_thput_accuracy_per_model()
            # This function is now deprecated, use utils.log_throughput_per_model
            # utils.log_thput_accuracy_per_model(rate_logger_per_model, i, requests_per_model,
            #                                    failed_per_model, accuracy_per_model)
        if done:
            break
        env.render()

        new_total_accuracy = env.simulator.total_accuracy
        new_total_successful = env.simulator.total_successful
        new_dropped = env.simulator.slo_timeouts['timeouts']
        new_late = env.simulator.slo_timeouts['late']
        estimated_throughput = env.simulator.ilp_stats['estimated_throughput']
        estimated_effective_accuracy = env.simulator.ilp_stats['estimated_effective_accuracy']
        ilp_utilization = env.simulator.ilp_stats['ilp_utilization']
        ilp_demand = env.simulator.ilp_stats['demand']
        proportional_lb = env.simulator.use_proportional
        demand_ewma = sum(env.simulator.ewma_demand.ravel())
        utils.log_throughput(logger=rate_logger, observation=observation,
                             simulation_time=i, allocation_window=allocation_window,
                             total_accuracy=new_total_accuracy-total_accuracy,
                             total_successful=new_total_successful-total_successful,
                             dropped=new_dropped-total_dropped,
                             late=new_late-total_late,
                             estimated_throughput=estimated_throughput,
                             estimated_effective_accuracy=estimated_effective_accuracy,
                             ilp_utilization=ilp_utilization,
                             proportional_lb=proportional_lb,
                             ilp_demand=ilp_demand,
                             demand_ewma=demand_ewma)
        total_accuracy = new_total_accuracy
        total_successful = new_total_successful
        total_dropped = new_dropped
        total_late = new_late

        # Logging per-model_family stats
        new_total_accuracy_per_model = env.simulator.total_accuracy_per_model
        new_total_successful_per_model = env.simulator.total_successful_per_model
        new_dropped_per_model = env.simulator.slo_timeouts_per_executor['timeouts']
        new_late_per_model = env.simulator.slo_timeouts_per_executor['late']

        window_total_accuracy_per_model = dict_subtraction(new_total_accuracy_per_model,
                                                           total_accuracy_per_model)
        window_total_successful_per_model = dict_subtraction(new_total_successful_per_model,
                                                             total_successful_per_model)
        window_dropped_per_model = dict_subtraction(new_dropped_per_model,
                                                    total_dropped_per_model)
        window_late_per_model = dict_subtraction(new_late_per_model, total_late_per_model)
        utils.log_throughput_per_model(logger=rate_logger_per_model,
                                       observation=observation,
                                       simulation_time=i,
                                       allocation_window=allocation_window,
                                       total_accuracy_per_model=window_total_accuracy_per_model,
                                       total_successful_per_model=window_total_successful_per_model,
                                       dropped_per_model=window_dropped_per_model,
                                       late_per_model=window_late_per_model,
                                       executor_to_idx=env.simulator.isi_to_idx
                                       )
        total_successful_per_model = copy.deepcopy(new_total_successful_per_model)
        total_accuracy_per_model = copy.deepcopy(new_total_accuracy_per_model)
        total_dropped_per_model = copy.deepcopy(new_dropped_per_model)
        total_late_per_model = copy.deepcopy(new_late_per_model)

    end = time.time()

    print()
    print('---------------------------')
    print('Printing overall statistics')
    print('---------------------------')
    print(f'Test time: {(end-start)} seconds')
    logfile.close()

    completed_requests = env.simulator.completed_requests
    sim_time_elapsed = env.simulator.clock / testing_steps
    overall_throughput = completed_requests / sim_time_elapsed

    total_accuracy = env.simulator.total_accuracy
    effective_accuracy = total_accuracy / env.simulator.total_successful

    bumped_succeeded = env.simulator.slo_timeouts['succeeded']
    bumped_failed = env.simulator.slo_timeouts['timeouts']
    total_bumped_failed_per_model = sum(list(env.simulator.slo_timeouts_per_executor['timeouts'].values()))
    bumped_total = bumped_failed + bumped_succeeded
    bumped_violation_ratio = bumped_failed / bumped_total
    bumped_late = env.simulator.slo_timeouts['late']
    bumped_late_ratio = bumped_late / bumped_total
    total_slo_violations = bumped_failed + bumped_late
    slo_violation_ratio = total_slo_violations / bumped_total

    print()
    print(f'Overall throughput (requests/sec): {overall_throughput}')
    print(f'Overall effective accuracy: {effective_accuracy}')
    print(f'Overall SLO violation ratio: {slo_violation_ratio}')
    print()

    print(f'Total requests: {bumped_total}')
    print(f'Dropped requests: {bumped_failed} dropped ratio: {bumped_violation_ratio}')
    print(f'Late requests: {bumped_late}, late ratio: {bumped_late_ratio}')
    print(f'Total SLO violations (dropped + late): {total_slo_violations}')

    print()
    print('---------------')
    print('Logging details')
    print('---------------')
    print(f'Aggregate throughput and accuracy logs written to: {rate_loggerfile}')
    print(f'Per-model throughput and accuracy logs written to: {rate_logger_per_model_file}')
    print(f'Latency logs written to: {env.simulator.latency_logfilename}')
    print()

if __name__=='__main__':
    main(getargs())
