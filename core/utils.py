import time
import scipy
import numpy as np


def log_throughput(logger, observation, simulation_time, allocation_window,
                   total_accuracy, total_successful, dropped, late,
                   estimated_throughput, estimated_effective_accuracy,
                   ilp_utilization, proportional_lb, ilp_demand,
                   demand_ewma):
    ''' Writes aggregate throughput and accuracy logs in the following CSV format:
    wallclock_time,simulation_time,demand,throughput,capacity
    '''
    demand = np.sum(observation[:, -2])
    failed_request_rate = np.sum(observation[:, -1])
    throughput = demand - failed_request_rate

    latency_matrix = observation[:-1, 4:8]
    throughput_matrix = 1000 / latency_matrix
    allocation_matrix = observation[:-1, 0:4]

    capacity_matrix = np.multiply(allocation_matrix, throughput_matrix)
    capacity = np.sum(capacity_matrix) * allocation_window / 1000

    if total_successful == 0:
        effective_accuracy = 0
    else:
        effective_accuracy = total_accuracy / total_successful
    
    logger.info(f'{time.time()},{simulation_time},{demand},{throughput},{capacity},'
                f'{effective_accuracy:.2f},{total_accuracy:.2f},{total_successful},'
                f'{dropped},{late},{estimated_throughput:.1f},'
                f'{estimated_effective_accuracy:.2f},{ilp_utilization:.2f},'
                f'{proportional_lb},{ilp_demand:.2f},{demand_ewma:.2f}')
    

def log_throughput_per_model(logger, observation, simulation_time, allocation_window,
                             total_accuracy_per_model, total_successful_per_model,
                             dropped_per_model, late_per_model, executor_to_idx):
    ''' Writes throughput and accuracy logs per model
    '''
    for model in total_successful_per_model:
        simulator_idx = executor_to_idx[model]

        demand = np.sum(observation[simulator_idx, -2])
        failed_request_rate = np.sum(observation[simulator_idx, -1])
        throughput = demand - failed_request_rate

        latency_matrix = observation[simulator_idx, 4:8]
        throughput_matrix = 1000 / latency_matrix
        allocation_matrix = observation[simulator_idx, 0:4]

        capacity_matrix = np.multiply(allocation_matrix, throughput_matrix)
        capacity = np.sum(capacity_matrix) * allocation_window / 1000

        total_successful = total_successful_per_model.get(model, 0)
        total_accuracy = total_accuracy_per_model.get(model, 0)
        dropped = dropped_per_model.get(model, 0)
        late = late_per_model.get(model, 0)

        if total_successful == 0:
            effective_accuracy = 0
        else:
            effective_accuracy = total_accuracy / total_successful
        
        logger.info(f'{time.time()},{simulation_time},{model},{demand},{throughput},'
                    f'{capacity},{effective_accuracy:.2f},{total_accuracy:.2f},'
                    f'{total_successful},{dropped},{late}'
                    )
    return


def log_thput_accuracy_per_model(logger, simulation_time, requests, failed, accuracy):
    ''' (DEPRECATED) Writes throughput and accuracy logs per model in the following
    CSV format:
    wallclock_time,simulation_time,demand_nth_model,throughput_nth_model,normalized_throughput_nth_model,accuracy_nth_model
    '''
    line = f'{time.time()},{simulation_time}'

    for model in range(len(requests)):
        throughput = requests[model] - failed[model]

        line += f',{requests[model]},{throughput},'

        if requests[model] > 0:
            normalized_throughput = throughput/requests[model]
        else:
            normalized_throughput = 0

        line += f'{normalized_throughput},{accuracy[model]}'

    logger.info(line)


def dict_subtraction(dict1, dict2):
    """ Returns dict1 - dict2 per key
    """
    res = {key: dict1[key] - dict2.get(key, 0) for key in dict1.keys()}
    return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
