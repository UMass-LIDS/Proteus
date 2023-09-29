import itertools
import logging
import math
import random
import time
import uuid
import numpy as np
from core.common import Behavior, TaskAssignment
from core.predictor import AccType, Predictor
from core.exceptions import ExecutorException


ACC_CONVERSION = {'CPU': 'CPU', 'GPU_AMPERE': 'GPU', 'VPU': 'VPU', 'GPU_PASCAL': 'FPGA'}


class Executor:
    def __init__(self, isi, task_assignment, logging_level, n_qos_levels=1,
                 behavior=Behavior.BESTEFFORT, runtimes=None, variant_runtimes=None,
                 variant_loadtimes=None, max_acc_per_type=0, simulator=None,
                 configured_max_batch_size=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)
        self.logging_level = logging_level

        self.id = uuid.uuid4().hex
        self.isi = isi
        self.n_qos_levels = n_qos_levels
        self.predictors = {}
        self.num_predictor_types = np.zeros(4 * n_qos_levels)
        self.max_acc_per_type = max_acc_per_type
        self.assigned_requests = {}
        self.iterator = itertools.cycle(self.predictors)
        self.behavior = behavior
        job_sched_algos = {'random': 1, 'round_robin': 2, 'eft_fifo': 3,
                           'lft_fifo': 4, 'infaas': 5, 'canary_routing': 6}
        self.task_assignment = TaskAssignment(job_sched_algos[task_assignment])
        self.runtimes = runtimes

        self.variant_runtimes = variant_runtimes
        self.variant_loadtimes = variant_loadtimes
        self.variant_accuracies = {}

        self.model_variants = {}
        self.variants_for_this_executor = []

        self.configured_max_batch_size = configured_max_batch_size

        # Mapping from model variant to percentage of requests to send to that variant
        self.canary_routing_table = {}

        self.simulator = simulator


    def add_predictor(self, acc_type=AccType.GPU, qos_level=0, variant_name=None):
        profiled_latencies = self.variant_runtimes[acc_type.value]
        
        if variant_name is None:
            min_accuracy = 100.0
            for candidate in self.model_variants[self.isi]:
                candidate_accuracy = self.variant_accuracies[(self.isi, candidate)]
                if candidate_accuracy < min_accuracy:
                    min_accuracy = candidate_accuracy
                    variant_name = candidate
        
        profiled_accuracy = self.variant_accuracies[(self.isi, variant_name)]

        predictor = Predictor(logging_level=self.logging_level, acc_type=acc_type.value,
                              qos_level=qos_level, profiled_accuracy=profiled_accuracy,
                              profiled_latencies=profiled_latencies, variant_name=variant_name,
                              executor=self, simulator=self.simulator,
                              configured_max_batch_size=self.configured_max_batch_size)
        self.predictors[predictor.id] = predictor
        self.num_predictor_types[acc_type.value-1 + qos_level*4] += 1
        self.iterator = itertools.cycle(self.predictors)

        return predictor.id


    def set_runtimes(self, runtimes=None):
        self.runtimes = runtimes

    
    def set_loadtimes(self, loadtimes=None):
        self.loadtimes = loadtimes

    
    def set_model_variants(self, model_variants={}):
        self.model_variants = model_variants
        self.variants_for_this_executor = model_variants[self.isi]


    def set_variant_accuracies(self, accuracies=None):
        self.variant_accuracies = accuracies

    
    def set_variant_runtimes(self, runtimes=None):
        self.variant_runtimes = runtimes

    
    def set_variant_loadtimes(self, loadtimes=None):
        self.variant_loadtimes = loadtimes

    
    def remove_predictor_by_id(self, id):
        if id in self.predictors:
            predictor_type = self.predictors[id].acc_type
            predictor_qos = self.predictors[id].qos_level
            self.num_predictor_types[predictor_type-1 + predictor_qos*4] -= 1
            del self.predictors[id]
            self.iterator = itertools.cycle(self.predictors)
            return True
        else:
            return False

    
    def remove_predictor_by_type(self, acc_type, qos_level=0):
        ''' If predictor of given type exists, remove it and return True.
            Otherwise, return False.
        '''
        for id in self.predictors:
            predictor_type = self.predictors[id].acc_type
            predictor_qos = self.predictors[id].qos_level
            if acc_type == predictor_type and qos_level == predictor_qos:
                self.num_predictor_types[predictor_type-1 + predictor_qos*4] -= 1
                del self.predictors[id]
                self.iterator = itertools.cycle(self.predictors)
                return True
        return False


    def process_request(self, event, clock, runtimes):
        if len(self.predictors) == 0:
            self.add_predictor()
            
        qos_met = True

        # Step 1: load balance

        # filter out predictors that match the request's QoS level
        filtered_predictors = list(filter(lambda key: self.predictors[key].qos_level >= event.qos_level, self.predictors))
        
        if len(filtered_predictors) == 0:
            qos_met = False
            self.log.debug('No predictors found for QoS level {} for request of {}'.format(event.qos_level, event.desc))
            if self.behavior == Behavior.BESTEFFORT:
                # choose from any QoS level
                filtered_predictors = list(self.predictors.keys())

        if self.task_assignment == TaskAssignment.RANDOM:
            predictor = self.predictors[random.choice(filtered_predictors)]
        elif self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME or self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
            finish_times = []
            for key in filtered_predictors:
                candidate = self.predictors[key]
                potential_runtime = candidate.profiled_latencies[(event.desc, candidate.variant_name, 1)]
                self.log.debug(f'busy till: {candidate.busy_till}, potential runtime: {potential_runtime}')
                if candidate.busy:
                    finish_time = candidate.busy_till + potential_runtime
                else:
                    finish_time = clock + potential_runtime

                # finish_time has to be within the request's deadline
                if finish_time > clock + event.deadline:
                    # set it to an invalid value
                    if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                        finish_time = np.inf
                    elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                        finish_time = -1
                finish_times.append(finish_time)
            self.log.debug('filtered_predictors: {}'.format(filtered_predictors))
            self.log.debug('finish_times: {}'.format(finish_times))
            if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                idx = finish_times.index(min(finish_times))
            elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                idx = finish_times.index(max(finish_times))
            self.log.debug('idx: {}'.format(idx))
            predictor = self.predictors[filtered_predictors[idx]]
            self.log.debug('filtered_predictors[idx]: {}'.format(filtered_predictors[idx]))
            self.log.debug('predictor: {}'.format(predictor))

        elif self.task_assignment == TaskAssignment.CANARY:
            predictor = None

            selected = random.choices(list(self.canary_routing_table.keys()),
                                    weights=list(self.canary_routing_table.values()),
                                    k=1)[0]
            (acc_type_unconvered, variant_name) = selected
            acc_type_converted = ACC_CONVERSION[acc_type_unconvered]
            selected = (acc_type_converted, variant_name)

            variants_dict = dict(filter(lambda x: (AccType(x[1].acc_type).name, x[1].variant_name) == selected,
                                        self.predictors.items()))
            variants = list(variants_dict.keys())

            if len(variants) == 0:
                # Instead of forcefully adding a predictor here, we should fail
                # this request instead
                self.simulator.bump_failed_request_stats(event)
                return

            self.log.debug(f'Variants: {variants}')
            selected_predictor_id = random.choice(variants)
            self.log.debug(f'Selected predictor id: {selected_predictor_id}')
            selected_predictor = self.predictors[selected_predictor_id]
            self.log.debug(f'Selected predictor: {selected_predictor}')
            predictor = selected_predictor


        elif self.task_assignment == TaskAssignment.INFAAS:
            accuracy_filtered_predictors = list(filter(lambda key: self.predictors[key].profiled_accuracy >= event.accuracy, self.predictors))
            predictor = None
            infaas_candidates = []
            not_found_reason = 'None'
            # There is atleast one predictor that matches the accuracy requirement of the request
            if len(accuracy_filtered_predictors) > 0:
                # If there is any predictor that can meet request
                for key in accuracy_filtered_predictors:
                    _predictor = self.predictors[key]
                    self.log.debug(f'infaas predictor profiled latencies: {_predictor.profiled_latencies}')
                    time.sleep(10)
                    peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                                                    event.qos_level)])
                    queued_requests = len(_predictor.request_dict)

                    self.log.debug(f'Throughput: {peak_throughput}')
                    self.log.debug(f'Queued requests: {queued_requests}')
                    if peak_throughput > queued_requests:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        infaas_candidates.append(_predictor)
                    else:
                        continue
                
                # We have now found a list of candidate predictors that match both accuracy
                # and deadline of the request, sorted by load
                infaas_candidates.sort(key=lambda x: x.load)

                if len(infaas_candidates) > 0:
                    # Select the least loaded candidate
                    predictor = infaas_candidates[0]
                    self.log.debug('found')
                else:
                    # There is no candidate that meets deadline
                    not_found_reason = 'latency_not_met'
                    self.log.debug('not found')
            else:
                # There is no predictor that even matches the accuracy requirement of the request
                not_found_reason = 'accuracy_not_met'
                self.log.debug('not found')
            
            # No predictor has been found yet, either because accuracy not met or deadline not met
            if predictor is None:
                # Now we try to find an inactive model variant that can meet accuracy+deadline
                isi_name = event.desc
                inactive_candidates = {}
                checked_variants = set(map(lambda key: self.predictors[key].variant_name, self.predictors))
                self.log.debug(f'checked variants: {checked_variants}')
                # time.sleep(1)

                for model_variant in self.model_variants[isi_name]:
                    # If we already checked for this variant
                    if model_variant in checked_variants:
                        continue
                    else:
                        for acc_type in AccType:
                            predictor_type = acc_type.value
                            runtime = self.variant_runtimes[predictor_type][(isi_name, model_variant, batch_size)]

                            self.log.debug(f'infaas, runtime: {runtime}, deadline: {event.deadline}')

                            largest_batch_size = self.get_largest_batch_size(model_variant=model_variant, acc_type=acc_type.value)

                            if math.isinf(runtime) or largest_batch_size == 0:
                                self.log.debug(f'largest_batch_size: {largest_batch_size}, runtime: {runtime}')
                                continue
                            
                            loadtime = self.variant_loadtimes[(isi_name, model_variant)]
                            total_time = runtime + loadtime

                            if total_time < event.deadline:
                                inactive_candidates[(model_variant, acc_type)] = total_time
                self.log.debug(f'inactive candidates: {inactive_candidates}')

                for candidate in inactive_candidates:
                    model_variant, acc_type = candidate

                    if self.num_predictor_types[acc_type.value-1 + event.qos_level*4] < self.max_acc_per_type:
                        predictor_id = self.add_predictor(acc_type=acc_type, variant_name=model_variant)
                        predictor = self.predictors[predictor_id]
                        self.log.debug(f'Predictor {predictor} added from inactive variants')
                        break

            # (Line 8) If we still cannot find one, we try to serve with the closest
            # possible accuracy and/or deadline
            if predictor is None:
                self.log.debug('No predictor found from inactive variants either')
                closest_acc_candidates = sorted(self.predictors, key=lambda x: abs(self.predictors[x].profiled_accuracy - event.accuracy))

                for candidate in closest_acc_candidates:
                    _predictor = self.predictors[candidate]
                    variant_name = _predictor.variant_name
                    acc_type = _predictor.acc_type

                    runtime = self.variant_runtimes[acc_type][(isi_name, variant_name, batch_size)]

                    peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                                                    _predictor.variant_name, batch_size)])
                    queued_requests = len(_predictor.request_dict)

                    self.log.debug('Throughput:', peak_throughput)
                    self.log.debug('Queued requests:', queued_requests)
                    if peak_throughput > queued_requests and runtime <= event.deadline:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        predictor = _predictor
                        self.log.debug('closest predictor {} found'.format(predictor))
                        break
                    else:
                        continue
            
        if predictor is None:
            # No predictor, there is literally nothing that can be done at this point except to drop request
            self.log.debug('No predictor available whatsoever')
            assigned = None
            qos_met = False
            return assigned, qos_met, 0

        else:
            # Predictor has been found, assign request to it

            # At this point, the variable 'predictor' should indicate which predictor has been
            # selected by one of the heuristics above

            # Step 2: read up runtime based on the predictor type selected
            batch_size = 1
            runtime = self.variant_runtimes[predictor.acc_type][(event.desc, predictor.variant_name, batch_size)]
            event.runtime = runtime

            # Step 3: assign to predictor selected
            assigned = predictor.assign_request(event, clock)
            accuracy = predictor.profiled_accuracy
            if assigned is not None:
                self.assigned_requests[event.id] = predictor
            else:
                self.log.debug('WARN: Request id {} for {} could not be assigned to any predictor. (Time: {})'.format(event.id, event.desc, clock))

            return assigned, qos_met, accuracy
    

    def trigger_infaas_upscaling(self):
        self.log.debug('infaas upscaling triggered')
        batch_size = 1
        for key in self.predictors:
            predictor = self.predictors[key]
            peak_throughput = 1000 / self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            queued_requests = len(predictor.request_dict)

            infaas_slack = self.simulator.infaas_slack

            if math.floor(peak_throughput * infaas_slack) > queued_requests:
                # no need for scaling
                self.log.debug('no need for scaling')
            else:
                self.log.debug(f'INFaaS autoscaling triggered at predictor {predictor.id}, '
                             f'executor {self.isi}')
                self.log.debug(f'floor(peak_throughput * infaas_slack): {(math.floor(peak_throughput * infaas_slack))}')
                self.log.debug(f'queued requests: {queued_requests}')

                # Now we have two options: (1) replication, (2) upgrading to meet SLO
                # Calculate the cost for both and choose the cheaper option

                # Option 1: Replication
                incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
                self.log.debug(f'incoming load: {incoming_load}')
                replicas_needed = math.ceil(incoming_load / (peak_throughput * infaas_slack))

                # Option 2: Upgrade
                # We might not upgrade to a model variant with higher accuracy, but we
                # could upgrade to a different accelerator type with higher throughput
                cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, predictor.variant_name, batch_size)]
                gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, predictor.variant_name, batch_size)]
                vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, predictor.variant_name, batch_size)]
                fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, predictor.variant_name, batch_size)]
                
                cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                cpu_available = self.simulator.available_predictors[0]
                gpu_available = self.simulator.available_predictors[1]
                vpu_available = self.simulator.available_predictors[2]
                fpga_available = self.simulator.available_predictors[3]

                if cpu_available == 0:
                    cpu_needed = math.inf
                if gpu_available == 0:
                    gpu_needed = math.inf
                if vpu_available == 0:
                    vpu_needed = math.inf
                if fpga_available == 0:
                    fpga_needed = math.inf

                # From INFaaS:
                # Each worker runs a model-autoscaler that approximates ILP as follows:
                # (a) Identify whether the constraints are in
                # danger of being violated,
                
                # (b) Consider two strategies, replicate
                # or upgrade/downgrade, to satisfy the constraints,

                # (c) Compute the objective for each of these scaling actions 
                # and pick the one that minimizes the objective cost function

                upgrade_needed = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])

                if upgrade_needed <= replicas_needed:
                    type_needed = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed)

                    self.log.debug(f'upgrade needed: {upgrade_needed}')
                    self.log.debug(f'type needed: {type_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and upgrade_needed > 0:
                        self.add_predictor(acc_type=AccType(type_needed+1))
                        upgrade_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    self.log.debug(f'upgrade needed: {upgrade_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    return
                else:
                    # add 'replicas_needed' predictors of type 'predictor.acc_type' or as many as possible
                    type_needed = predictor.acc_type
                    self.log.debug(f'replicas needed: {replicas_needed}')
                    self.log.debug(f'type needed: {type_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and replicas_needed > 0:
                        self.add_predictor(acc_type=type_needed)
                        replicas_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    self.log.debug(f'replicas needed: {replicas_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    return

    
    def trigger_infaas_v2_upscaling(self):
        # # From the INFaaS paper:
        # # Each worker runs a model-autoscaler that approximates ILP as follows:
        # # (a) Identify whether the constraints are in
        # # danger of being violated,
        
        # # (b) Consider two strategies, replicate
        # # or upgrade/downgrade, to satisfy the constraints,

        # # (c) Compute the objective for each of these scaling actions 
        # # and pick the one that minimizes the objective cost function

        self.log.debug('infaas v2 upscaling triggered')
        infaas_slack = self.simulator.infaas_slack

        incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
        total_peak_throughput = 0
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            if batch_size == 0:
                runtime = math.inf
                peak_throughput = 0
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
                peak_throughput = predictor.get_largest_batch_size() * 1000 / runtime
            total_peak_throughput += peak_throughput


        if math.floor(total_peak_throughput * infaas_slack) >= incoming_load:
            self.log.debug(f'no need for upscaling, incoming load: {incoming_load}')
            return

        predictors_to_add = []
        predictors_to_remove = []
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            if batch_size == 0:
                runtime = math.inf
                peak_throughput = 0
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
                peak_throughput = predictor.get_largest_batch_size() * 1000 / runtime
            queued_requests = len(predictor.request_dict)

            self.log.debug(f'INFaaS autoscaling triggered at predictor {predictor.id}, '
                            f'executor {self.isi}')
            self.log.debug(f'peak_throughput: {peak_throughput}')
            self.log.debug(f'floor(peak_throughput * infaas_slack): {(math.floor(peak_throughput * infaas_slack))}')
            self.log.debug(f'queued requests: {queued_requests}')

            # First we consider batch size increase, which costs 0 to change
            # Technically it is an upgrade, but we can increase it first and
            # see if anything still needs to be changed
            for batch_size in self.simulator.allowed_batch_sizes:
                if batch_size > predictor.get_largest_batch_size():
                    break
                else:
                    predictor.set_infaas_batch_size(batch_size)
            
            old_predictor_throughput = peak_throughput
            peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime
            predictor_peak_throughput = peak_throughput

            total_peak_throughput = total_peak_throughput - old_predictor_throughput + predictor_peak_throughput

            # Now we have two options: (1) replication, (2) upgrading to meet SLO

            # For option 1 and 2, calculate the cost for both and choose the cheaper option

            # Option 1: Replication
            incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
            self.log.debug(f'incoming load: {incoming_load}')
            if peak_throughput == 0:
                replicas_needed = math.inf
            else:
                replicas_needed = math.ceil(incoming_load / (peak_throughput * infaas_slack))

            # Replicating this given predictor will have as much cost as the
            # cost of this predictor, i.e., the accuracy drop of this predictor

            self.log.debug(f'replicas needed: {replicas_needed}, accuracy drop of '
                            f'this variant: {predictor.get_infaas_cost()}')
            # The cost for all replicas of this predictor is the same
            cost_replication = predictor.get_infaas_cost() * replicas_needed
            self.log.debug(f'total cost of replication: {cost_replication}')

            # Option 2: Upgrade
            # Upgrade could also mean upgrading to a higher batch size,
            # or upgrading to a model variant with different/higher accuracy

            # Option 2a: Change to a different model variant
            # Option 2b: Change to a different hardware accelerator and make multiple copies

            # Consider the different model variants
            # Based on Section 4.2.2, "Scaling up algorithm", it seems we have to
            # run (2b) for every model variant of this model family that can meet
            # SLO and support a higher throughput than the currently running variant
            # We calculate the cost_upgrade for each of these variants, then take
            # the minimum cost_upgrade and compare that with cost_replication, to take
            # the appropriate action

            model_variants = dict(filter(lambda x: x[0][0]==self.isi, predictor.profiled_latencies.items()))
            self.log.debug(f'self.isi: {self.isi}, model_variants: {model_variants}')
            cost_upgrade = np.inf
            upgrade_needed = None
            type_needed = None
            selected_variant = None
            for model_variant in model_variants:
                variant_name = model_variant[1]
                batch_size = model_variant[2]

                cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, variant_name, batch_size)]
                gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, variant_name, batch_size)]
                vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, variant_name, batch_size)]
                fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, variant_name, batch_size)]
                
                cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                cpu_available = self.simulator.available_predictors[0]
                gpu_available = self.simulator.available_predictors[1]
                vpu_available = self.simulator.available_predictors[2]
                fpga_available = self.simulator.available_predictors[3]

                if cpu_available == 0:
                    cpu_needed = math.inf
                if gpu_available == 0:
                    gpu_needed = math.inf
                if vpu_available == 0:
                    vpu_needed = math.inf
                if fpga_available == 0:
                    fpga_needed = math.inf

                upgrade_needed_variant = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])
                type_needed_variant = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed_variant)
                cost_upgrade_variant = self.model_variant_infaas_cost(variant_name) * upgrade_needed_variant
                self.log.debug(f'variant: {model_variant}, upgrade_needed_variant: {upgrade_needed_variant},' 
                                f' type_needed_variant: {type_needed_variant}, cost per instance: '
                                f'{self.model_variant_infaas_cost(variant_name)}')

                if cost_upgrade_variant < cost_upgrade:
                    cost_upgrade = cost_upgrade_variant
                    upgrade_needed = upgrade_needed_variant
                    type_needed = type_needed_variant
                    selected_variant = variant_name
            if type_needed is None:
                continue
            self.log.debug(f'selected variant for upgrade: {selected_variant}, cost '
                            f'of upgrade: {cost_upgrade}')

            self.log.debug(f'upgrade of type {type_needed} needed: {upgrade_needed},'
                            f'accuracy drop of this variant: {predictor.get_infaas_cost()}')
            self.log.debug(f'total cost of upgrade: {cost_upgrade}')

            if cost_upgrade <= cost_replication:
                self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                while self.simulator.available_predictors[type_needed] > 0 and upgrade_needed > 0:
                    self.add_predictor(acc_type=AccType(type_needed+1), variant_name=selected_variant)
                    upgrade_needed -= 1
                    self.simulator.available_predictors[type_needed] -= 1
                    print(f'infaas upgraded predictor of type: {type_needed}, available predictors: {self.simulator.available_predictors}')
                self.log.debug(f'upgrade needed: {upgrade_needed}')
                self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                return
            else:
                # add 'replicas_needed' predictors of type 'predictor.acc_type' or as many as possible
                type_needed = predictor.acc_type - 1
                self.log.debug(f'replicas needed: {replicas_needed}')
                self.log.debug(f'type needed: {type_needed}')
                self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                while self.simulator.available_predictors[type_needed] > 0 and replicas_needed > 0:
                    self.add_predictor(acc_type=AccType(type_needed+1))
                    replicas_needed -= 1
                    self.simulator.available_predictors[type_needed] -= 1
                    print(f'infaas replicated predictor of type: {type_needed}, available predictors: {self.simulator.available_predictors}')
                self.log.debug(f'replicas needed: {upgrade_needed}')
                self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                return
        
        # We can't remove predictors inside the above loop since it will
        # change the size of the loop
        for id in predictors_to_remove:
            predictor_type = self.predictors[id].acc_type
            removed = self.remove_predictor_by_id(id)
            if removed:
                self.simulator.available_predictors[predictor_type-1] += 1

        for tuple in predictors_to_add:
            acc_type, variant_name = tuple
            self.add_predictor(acc_type=acc_type, variant_name=variant_name)
            self.simulator.available_predictors[acc_type.value-1] -= 1

        self.log.debug(f'predictors_to_add: {predictors_to_add}')
        self.log.debug(f'predictors_to_remove: {predictors_to_remove}')
        
        return
                
    
    def trigger_infaas_v3_upscaling(self):
        # # From the INFaaS paper:
        # # Each worker runs a model-autoscaler that approximates ILP as follows:
        # # (a) Identify whether the constraints are in
        # # danger of being violated,
        
        # # (b) Consider two strategies, replicate
        # # or upgrade/downgrade, to satisfy the constraints,

        # # (c) Compute the objective for each of these scaling actions 
        # # and pick the one that minimizes the objective cost function

        self.log.debug('infaas v2 upscaling triggered')

        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            if batch_size == 0:
                runtime = math.inf
                peak_throughput = 0
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
                peak_throughput = batch_size * 1000 / runtime
            queued_requests = len(predictor.request_dict)

            infaas_slack = self.simulator.infaas_slack

            if math.floor(peak_throughput * infaas_slack) >= queued_requests:
                # no need for scaling
                self.log.debug(f'no need for upscaling, queued_requests: {queued_requests}')
            else:
                self.log.debug(f'INFaaS autoscaling triggered at predictor {predictor.id}, '
                             f'executor {self.isi}')
                self.log.debug(f'peak_throughput: {peak_throughput}')
                self.log.debug(f'floor(peak_throughput * infaas_slack): {(math.floor(peak_throughput * infaas_slack))}')
                self.log.debug(f'queued requests: {queued_requests}')

                # First we consider batch size increase, which costs 0 to change
                # Technically it is an upgrade, but we can increase it first and
                # see if anything still needs to be changed
                for batch_size in self.simulator.allowed_batch_sizes:
                    if batch_size > predictor.get_largest_batch_size():
                        break
                    else:
                        predictor.set_infaas_batch_size(batch_size)
                
                peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime

                # Now we have two options: (1) replication, (2) upgrading to meet SLO
                # Calculate the cost for both and choose the cheaper option

                # Option 1: Replication
                incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
                self.log.debug(f'incoming load: {incoming_load}')
                if peak_throughput == 0:
                    replicas_needed = math.inf
                else:
                    replicas_needed = math.ceil(incoming_load / (peak_throughput * infaas_slack))

                # Replicating this given predictor will have as much cost as the
                # cost of this predictor, i.e., the accuracy drop of this predictor

                self.log.debug(f'replicas needed: {replicas_needed}, accuracy drop of '
                             f'this variant: {predictor.get_infaas_cost()}')
                # The cost for all replicas of this predictor is the same
                cost_replication = predictor.get_infaas_cost() * replicas_needed
                self.log.debug(f'total cost of replication: {cost_replication}')

                # Option 2: Upgrade
                # Upgrade could also mean upgrading to a higher batch size,
                # or upgrading to a model variant with different/higher accuracy

                # Option 2a: Change to a different model variant
                # Option 2b: Change to a different hardware accelerator and make multiple copies

                # Consider the different model variants
                # Based on Section 4.2.2, "Scaling up algorithm", it seems we have to
                # run (2b) for every model variant of this model family that can meet
                # SLO and support a higher throughput than the currently running variant
                # We calculate the cost_upgrade for each of these variants, then take
                # the minimum cost_upgrade and compare that with cost_replication, to take
                # the appropriate action

                model_variants = dict(filter(lambda x: x[0][0]==self.isi, predictor.profiled_latencies.items()))
                self.log.debug(f'self.isi: {self.isi}, model_variants: {model_variants}')
                cost_upgrade = np.inf
                upgrade_needed = None
                type_needed = None
                selected_variant = None
                for model_variant in model_variants:
                    variant_name = model_variant[1]
                    batch_size = model_variant[2]

                    cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, variant_name, batch_size)]
                    gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, variant_name, batch_size)]
                    vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, variant_name, batch_size)]
                    fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, variant_name, batch_size)]
                    
                    cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                    gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                    vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                    fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                    cpu_available = self.simulator.available_predictors[0]
                    gpu_available = self.simulator.available_predictors[1]
                    vpu_available = self.simulator.available_predictors[2]
                    fpga_available = self.simulator.available_predictors[3]

                    if cpu_available == 0:
                        cpu_needed = math.inf
                    if gpu_available == 0:
                        gpu_needed = math.inf
                    if vpu_available == 0:
                        vpu_needed = math.inf
                    if fpga_available == 0:
                        fpga_needed = math.inf

                    upgrade_needed_variant = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])
                    type_needed_variant = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed_variant)
                    cost_upgrade_variant = self.model_variant_infaas_cost(variant_name) * upgrade_needed_variant
                    self.log.debug(f'variant: {model_variant}, upgrade_needed_variant: {upgrade_needed_variant},' 
                                 f' type_needed_variant: {type_needed_variant}, cost per instance: '
                                 f'{self.model_variant_infaas_cost(variant_name)}')

                    if cost_upgrade_variant < cost_upgrade:
                        cost_upgrade = cost_upgrade_variant
                        upgrade_needed = upgrade_needed_variant
                        type_needed = type_needed_variant
                        selected_variant = variant_name
                if type_needed is None:
                    continue
                self.log.debug(f'selected variant for upgrade: {selected_variant}, cost '
                              f'of upgrade: {cost_upgrade}')

                self.log.debug(f'upgrade of type {type_needed} needed: {upgrade_needed},'
                             f'accuracy drop of this variant: {predictor.get_infaas_cost()}')
                self.log.debug(f'total cost of upgrade: {cost_upgrade}')

                if cost_upgrade <= cost_replication:
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and upgrade_needed > 0:
                        self.add_predictor(acc_type=AccType(type_needed+1), variant_name=selected_variant)
                        upgrade_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                        print(f'infaas upgraded predictor of type: {type_needed}, available predictors: {self.simulator.available_predictors}')
                    self.log.debug(f'upgrade needed: {upgrade_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    return
                else:
                    # add 'replicas_needed' predictors of type 'predictor.acc_type' or as many as possible
                    type_needed = predictor.acc_type - 1
                    self.log.debug(f'replicas needed: {replicas_needed}')
                    self.log.debug(f'type needed: {type_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and replicas_needed > 0:
                        self.add_predictor(acc_type=AccType(type_needed+1))
                        replicas_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                        print(f'infaas replicated predictor of type: {type_needed}, available predictors: {self.simulator.available_predictors}')
                    self.log.debug(f'replicas needed: {upgrade_needed}')
                    self.log.debug(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    return
        return

    
    def trigger_infaas_downscaling(self):
        self.log.error('infaas v1 downscaling not implemented, use v2')
    

    def trigger_infaas_v2_downscaling(self):
        self.log.debug('infaas v2 downscaling triggered')
        infaas_slack = self.simulator.infaas_slack

        # First find the total load and the capacity of the system
        total_peak_throughput = 0
        total_queued_requests = 0
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            if batch_size == 0:
                runtime = math.inf
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime
            total_peak_throughput += predictor_peak_throughput

            queued_requests = len(predictor.request_dict)
            total_queued_requests += queued_requests
        
        # Now see if any predictors can be removed or downgraded
        predictors_to_remove = []
        predictors_to_add = []
        for key in self.predictors:
            predictor = self.predictors[key]
            if batch_size == 0:
                runtime = math.inf
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime

            # Two options, (i) removing, (ii) downgrading

            # (i) Removing: If load can be met after removing this predictor
            new_throughput = total_peak_throughput - predictor_peak_throughput
            if new_throughput * infaas_slack > total_queued_requests:
                # remove this predictor
                predictors_to_remove.append(predictor.id)
                total_peak_throughput = new_throughput
                continue

            # (ii) Downgrading: Consider downgrade options
            downgraded = False

            # We could downgrade to smaller batch size, or running on different hardware

            incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]

            # First consider downgrading accuracy
            # If throughput improves by downgrading accuracy, downgrade
            self.log.debug(f'self.variant_accuracies: {self.variant_accuracies}')
            executor_name = self.isi
            available_variants = list(filter(lambda x: x[0] == executor_name, self.variant_accuracies))
            self.log.debug(f'available_variants: {available_variants}')

            max_accuracy = 0
            min_accuracy = math.inf
            min_accuracy_variant = None
            for availabe_variant in available_variants:
                max_accuracy = max(max_accuracy, self.variant_accuracies[availabe_variant])
                if self.variant_accuracies[availabe_variant] < min_accuracy:
                    min_accuracy = self.variant_accuracies[availabe_variant]
                    min_accuracy_variant = availabe_variant

            variant_throughputs = {}
            variant_costs = {}
            variant_throughput_gaps = {}
            for availabe_variant in available_variants:
                (isi, availabe_variant_name) = availabe_variant
                batch_size = self.get_largest_batch_size(availabe_variant_name, predictor.acc_type)
                if batch_size == 0:
                    variant_latency = math.inf
                    variant_throughput = 0
                else:
                    variant_latency = self.variant_runtimes[predictor.acc_type][(isi, availabe_variant_name, batch_size)]
                    variant_throughput = batch_size * 1000 / variant_latency

                variant_throughputs[availabe_variant] = variant_throughput
                accuracy_drop = max_accuracy - self.variant_accuracies[availabe_variant]
                variant_costs[availabe_variant] = accuracy_drop

                proposed_throughput = total_peak_throughput - predictor_peak_throughput + variant_throughput
                proposed_throughput_gap = max(0, incoming_load - proposed_throughput * infaas_slack * self.simulator.infaas_downscale_slack)
                variant_throughput_gaps[availabe_variant] = proposed_throughput_gap
            self.log.debug(f'variant_throughputs: {variant_throughputs}')
            self.log.debug(f'variant_costs: {variant_costs}')
            self.log.debug(f'variant_throughput_gaps: {variant_throughput_gaps}')

            # Go through all variants of this, if there is one that can meet throughput, select that,
            # otherwise, select the one with least accuracy (highest throughput)
            selected_variant = None
            least_cost_found = math.inf
            
            # If there is no variant that can meet throughput, select one with least accuracy (highest throuhgput)
            if selected_variant is None:
                selected_variant = min_accuracy_variant
            
            (isi, selected_variant_name) = selected_variant
            if selected_variant_name != predictor.variant_name:
                predictors_to_remove.append(predictor.id)
                predictors_to_add.append((AccType(predictor.acc_type), selected_variant_name))
                old_throughput = total_peak_throughput
                total_peak_throughput = total_peak_throughput - predictor_peak_throughput + variant_throughputs[selected_variant]
                new_accuracy = self.variant_accuracies[selected_variant]
                old_accuracy = self.variant_accuracies[(self.isi, predictor.variant_name)]
                self.log.debug(f'Found a variant with lower accuracy to replace one with higher accuracy, '
                               f'new (accuracy, variant): ({new_accuracy},{selected_variant_name}), '
                               f'old (accuracy, variant: ({old_accuracy},{predictor.variant_name}), '
                               f'old peak throughput: {old_throughput}, new peak throughput: {total_peak_throughput}')
                continue
            else:
                self.log.debug(f'Already running variant with lowest accuracy, checking other options for downgrading..')


            if total_peak_throughput * infaas_slack * self.simulator.infaas_downscale_slack <= incoming_load:
                self.log.debug(f'no need to downscale')
                return
            
            # Then consider downgrading to smaller batch size, going in increasing order
            for batch_size in self.simulator.allowed_batch_sizes:
                if batch_size > predictor.get_largest_batch_size():
                    break

                batch_peak_throughput = batch_size * 1000 / runtime
                new_throughput = total_peak_throughput - predictor_peak_throughput + batch_peak_throughput
                if new_throughput * infaas_slack > total_queued_requests:
                    predictor.set_infaas_batch_size(batch_size)
                    total_peak_throughput = new_throughput
                    downgraded = True
                    break
            
            # If we have already downgraded, no need to consider second option
            # for downgrading
            if downgraded:
                continue

            # Now consider running on different hardware
            if predictor.acc_type == AccType.CPU or predictor.acc_type == AccType.VPU:
                # It cannot be downgraded further as CPU/VPU are the slowest
                continue
            else:
                # We are running on GPU or FPGA (in our setting, FPGA is also GPU)
                # If CPU or VPU is available, and load can be supported through that,
                # downgrade
                cpu_available = self.simulator.available_predictors[0]
                vpu_available = self.simulator.available_predictors[2]

                if self.get_largest_batch_size(predictor.variant_name, 'CPU') == 0:
                    continue

                if cpu_available > 0 or vpu_available > 0:
                    if predictor.get_largest_batch_size() == 0:
                        cpu_peak_throughput = 0
                    else:
                        cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi,
                                                    predictor.variant_name,
                                                    predictor.get_infaas_batch_size())]
                    new_throughput = total_peak_throughput - predictor_peak_throughput + cpu_peak_throughput

                    if new_throughput * infaas_slack > total_queued_requests:
                        self.log.debug(f'removing gpu/fpga predictor and adding cpu/vpu predictor'
                              f' for {predictor.executor.isi}')
                        total_peak_throughput = new_throughput
                        # remove this predictor
                        predictors_to_remove.append(predictor.id)
                        # add a CPU or VPU predictor, whichever is available
                        if cpu_available > 0:
                            predictors_to_add.append((AccType.CPU, predictor.variant_name))
                        elif vpu_available > 0:
                            predictors_to_add.append((AccType.VPU, predictor.variant_name))
                        else:
                            self.log.error('infaas_v2_downscaling: neither CPU nor VPU is available!')
                            time.sleep(1)
        
        # We can't add/remove predictors inside the above loop since it will
        # change the size of the loop
        for id in predictors_to_remove:
            predictor_type = self.predictors[id].acc_type
            removed = self.remove_predictor_by_id(id)
            if removed:
                self.simulator.available_predictors[predictor_type-1] += 1

        for tuple in predictors_to_add:
            acc_type, variant_name = tuple
            self.add_predictor(acc_type=acc_type, variant_name=variant_name)
            self.simulator.available_predictors[acc_type.value-1] -= 1

        return
    

    def trigger_infaas_v3_downscaling(self):
        self.log.debug('infaas v2 downscaling triggered')
        infaas_slack = self.simulator.infaas_slack

        # First find the total load and the capacity of the system
        total_peak_throughput = 0
        total_queued_requests = 0
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            if batch_size == 0:
                runtime = math.inf
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime
            total_peak_throughput += predictor_peak_throughput

            queued_requests = len(predictor.request_dict)
            total_queued_requests += queued_requests
        
        # Now see if any predictors can be removed or downgraded
        predictors_to_remove = []
        predictors_to_add = []
        for key in self.predictors:
            predictor = self.predictors[key]
            if batch_size == 0:
                runtime = math.inf
            else:
                runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime

            # Two options, (i) removing, (ii) downgrading

            # (i) Removing: If load can be met after removing this predictor
            new_throughput = total_peak_throughput - predictor_peak_throughput
            if new_throughput * infaas_slack > total_queued_requests:
                # remove this predictor
                predictors_to_remove.append(predictor.id)
                total_peak_throughput = new_throughput
                continue

            # (ii) Downgrading: Consider downgrade options
            downgraded = False

            # We could downgrade to smaller batch size, or running on different hardware
            
            # First consider smaller batch sizes, going in increasing order
            for batch_size in self.simulator.allowed_batch_sizes:
                if batch_size > predictor.get_largest_batch_size():
                    break

                batch_peak_throughput = batch_size * 1000 / runtime
                new_throughput = total_peak_throughput - predictor_peak_throughput + batch_peak_throughput
                if new_throughput * infaas_slack > total_queued_requests:
                    predictor.set_infaas_batch_size(batch_size)
                    total_peak_throughput = new_throughput
                    downgraded = True
                    break
            
            # If we have already downgraded, no need to consider second option
            # for downgrading
            if downgraded:
                continue

            # Now consider running on different hardware
            if predictor.acc_type == AccType.CPU or predictor.acc_type == AccType.VPU:
                # It cannot be downgraded further as CPU/VPU are the slowest
                continue
            else:
                # We are running on GPU or FPGA (in our setting, FPGA is also GPU)
                # If CPU or VPU is available, and load can be supported through that,
                # downgrade
                cpu_available = self.simulator.available_predictors[0]
                vpu_available = self.simulator.available_predictors[2]

                if self.get_largest_batch_size(predictor.variant_name, 'CPU') == 0:
                    continue

                if cpu_available > 0 or vpu_available > 0:
                    cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi,
                                                predictor.variant_name,
                                                predictor.get_infaas_batch_size())]
                    new_throughput = total_peak_throughput - predictor_peak_throughput + cpu_peak_throughput

                    if new_throughput * infaas_slack > total_queued_requests:
                        self.log.debug(f'removing gpu/fpga predictor and adding cpu/vpu predictor'
                              f' for {predictor.executor.isi}')
                        total_peak_throughput = new_throughput
                        # remove this predictor
                        predictors_to_remove.append(predictor.id)
                        # add a CPU or VPU predictor, whichever is available
                        if cpu_available > 0:
                            predictors_to_add.append((AccType.CPU, predictor.variant_name))
                        elif vpu_available > 0:
                            predictors_to_add.append((AccType.VPU, predictor.variant_name))
                        else:
                            self.log.error('infaas_v2_downscaling: neither CPU nor VPU is available!')
                            time.sleep(1)
        
        # We can't add/remove predictors inside the above loop since it will
        # change the size of the loop
        for id in predictors_to_remove:
            self.remove_predictor_by_id(id)

        for tuple in predictors_to_add:
            acc_type, variant_name = tuple
            self.add_predictor(acc_type=acc_type, variant_name=variant_name)

        return

    
    def model_variant_infaas_cost(self, model_variant):
        ''' Returns the accuracy cost of a model variant
        '''
        isi = self.simulator.get_isi_from_variant_name(model_variant)
        self.log.debug(f'model_variant_infaas_cost, isi: {isi}, model_variant: '
                      f'{model_variant}, self.variant_accuracies: {self.variant_accuracies}')
        model_variants = dict(filter(lambda x: x[0][0]==isi, self.variant_accuracies.items()))
        self.log.debug(f'model_variants: {model_variants}')
        max_accuracy = max(model_variants.values())
        self.log.debug(f'max accuracy: {max_accuracy}')
        cost = max_accuracy - model_variants[(isi, model_variant)]
        self.log.debug(f'cost: {cost}')
        return cost

    
    def finish_request(self, event, clock):
        if event.id not in self.assigned_requests:
            self.log.warn('ERROR: Could not find assigned request.')
            time.sleep(1)
            return False
        
        predictor = self.assigned_requests[event.id]

        finished = predictor.finish_request(event)
        if finished:
            del self.assigned_requests[event.id]
            self.log.debug('Finished request for {} at predictor {}. (Time: {})'.format(event.desc, predictor.id, clock))
            return True
        else:
            self.log.debug('WARN: Could not finish request at predictor {}, \
                    executor {}. (Time: {})'.format(predictor.id, self.id, clock))
            return False

    
    def enqueue_request(self, event, clock):
        ''' When batching is enabled, we send an incoming request to the queue of an
        appropriate predictor. The predictor will dequeue the request and process it
        with a batch of requests.
        '''
        self.log.debug(f'enqueuing request for isi: {event.desc}, arrived at: {self.isi}, id: {self.id}')

        if self.task_assignment == TaskAssignment.CANARY:
            if len(self.canary_routing_table) == 0:
                self.simulator.bump_failed_request_stats(event)
                self.log.debug('failed request because routing table is empty')
                return
            
            if self.simulator.model_assignment == 'ilp' and self.simulator.use_proportional == True:
                selected = None
                weights = list(map(lambda x: x.peak_throughput, self.predictors.values()))
                weights = np.ones(len(self.predictors))
                self.log.debug(f'weights: {weights}')
                if not(any(x > 0 for x in weights)):
                    self.simulator.bump_failed_request_stats(event)
                    self.log.warn('failed request because no predictor has throughput more than 0')
                    return
                selected = random.choices(list(self.predictors.values()),
                                          weights=weights,
                                          k=1)[0]
                acc_type_unconverted = AccType(selected.acc_type).name
                acc_type_converted = acc_type_unconverted
                variant_name = selected.variant_name
                selected = (acc_type_converted, variant_name)

            else:
                selected = random.choices(list(self.canary_routing_table.keys()),
                                        weights=list(self.canary_routing_table.values()),
                                        k=1)[0]
                (acc_type_unconverted, variant_name) = selected
                acc_type_converted = ACC_CONVERSION[acc_type_unconverted]
                selected = (acc_type_converted, variant_name)

            variants_dict = dict(filter(lambda x: (AccType(x[1].acc_type).name, x[1].variant_name) == selected,
                                        self.predictors.items()))
            variants = list(variants_dict.keys())

            if len(variants) == 0:
                # Instead of forcefully adding a predictor here, we should fail this
                # request instead
                self.simulator.bump_failed_request_stats(event)
                return

            self.log.debug(f'Variants: {variants}')
            selected_predictor_id = random.choice(variants)
            self.log.debug(f'Selected predictor id: {selected_predictor_id}')
            selected_predictor = self.predictors[selected_predictor_id]
            self.log.debug(f'Selected predictor: {selected_predictor}')

            selected_predictor.enqueue_request(event, clock)
            self.assigned_requests[event.id] = selected_predictor
            return
        
        elif self.task_assignment == TaskAssignment.INFAAS:
            self.log.debug(f'Selecting predictor for batch request processing with INFaaS')

            accuracy_filtered_predictors = list(filter(lambda key: self.predictors[key].profiled_accuracy >= event.accuracy, self.predictors))

            predictor = None
            infaas_candidates = []
            not_found_reason = 'None'

            # There is atleast one predictor that matches the accuracy requirement of the request
            if len(accuracy_filtered_predictors) > 0:
                # If there is any predictor that can serve request
                for key in accuracy_filtered_predictors:
                    _predictor = self.predictors[key]
                    batch_size = _predictor.get_infaas_batch_size()
                    executor_isi = _predictor.executor.isi
                    if batch_size == 0:
                        profiled_latency = math.inf
                        peak_throughput = 0
                    else:
                        profiled_latency = _predictor.profiled_latencies[(executor_isi, _predictor.variant_name, batch_size)]
                        peak_throughput = math.floor(batch_size * 1000 / profiled_latency)
                    queued_requests = len(_predictor.request_dict)

                    self.log.debug(f'Throughput: {peak_throughput}')
                    self.log.debug(f'Queued requests: {queued_requests}')
                    if peak_throughput > queued_requests:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        infaas_candidates.append(_predictor)
                    else:
                        continue

                if len(infaas_candidates) > 0:
                    # We have now found a list of candidate predictors that match
                    # both accuracy and deadline of the request, sorted by load
                    infaas_candidates.sort(key=lambda x: x.load)

                    # Select the least loaded candidate
                    predictor = infaas_candidates[0]
                    self.log.debug(f'predictor found: {predictor}')
                else:
                    # There is no candidate that meets deadline
                    not_found_reason = 'latency_not_met'
                    self.log.debug(f'predictor not found because latency SLO not met')
            else:
                # There is no predictor that even matches the accuracy requirement of the request
                not_found_reason = 'accuracy_not_met'
                self.log.debug(f'predictor not found because accuracy not met')
            
            # No predictor has been found yet, either because accuracy not met or deadline not met
            if predictor is None:
                # Now we try to find an inactive model variant that can meet accuracy+deadline
                isi_name = event.desc
                inactive_candidates = {}
                checked_variants = set(map(lambda key: self.predictors[key].variant_name, self.predictors))
                self.log.debug(f'checked variants: {checked_variants}')

                for model_variant in self.model_variants[isi_name]:
                    # If we already checked for this variant
                    if model_variant in checked_variants:
                        continue
                    else:
                        for acc_type in AccType:
                            predictor_type = acc_type.value
                            largest_batch_size = self.get_largest_batch_size(model_variant=model_variant, acc_type=acc_type.value)
                            
                            if largest_batch_size == 0:
                                largest_batch_size = 1

                            runtime = self.variant_runtimes[predictor_type][(isi_name, model_variant, largest_batch_size)]

                            self.log.debug(f'infaas, runtime: {runtime}, deadline: {event.deadline}')
                            
                            loadtime = self.variant_loadtimes[(isi_name, model_variant)]
                            total_time = runtime + loadtime

                            if total_time < event.deadline:
                                inactive_candidates[(model_variant, acc_type)] = total_time
                self.log.debug(f'inactive candidates: {inactive_candidates}')

                for candidate in inactive_candidates:
                    model_variant, acc_type = candidate

                    if self.num_predictor_types[acc_type.value-1 + event.qos_level*4] < self.max_acc_per_type and self.simulator.available_predictors[acc_type.value-1] > 0:
                        predictor_id = self.add_predictor(acc_type=acc_type, variant_name=model_variant)
                        self.simulator.available_predictors[acc_type.value-1] -= 1
                        predictor = self.predictors[predictor_id]
                        self.log.debug(f'Predictor {predictor} added from inactive variants')
                        break

            # (Line 8) If we still cannot find one, we try to serve with the closest
            # possible accuracy and/or deadline
            if predictor is None:
                self.log.debug('No predictor found from inactive variants either')
                closest_acc_candidates = sorted(self.predictors, key=lambda x: abs(self.predictors[x].profiled_accuracy - event.accuracy))

                for candidate in closest_acc_candidates:
                    _predictor = self.predictors[candidate]
                    variant_name = _predictor.variant_name
                    acc_type = _predictor.acc_type
                    batch_size = _predictor.get_infaas_batch_size()

                    if batch_size == 0:
                        batch_size = 1

                    runtime = self.variant_runtimes[acc_type][(isi_name, variant_name, batch_size)]

                    peak_throughput = math.floor(batch_size * 1000 / runtime)
                    queued_requests = len(_predictor.request_dict)

                    self.log.debug(f'Throughput: {peak_throughput}')
                    self.log.debug(f'Queued requests: {queued_requests}')
                    if peak_throughput > queued_requests and runtime <= event.deadline:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        predictor = _predictor
                        self.log.debug(f'closest predictor {predictor} found')
                        break
                    else:
                        continue

            if predictor is None:
                self.log.debug(f'infaas: predictor not found. not sure what to do here')
                closest_acc_candidates = sorted(self.predictors, key=lambda x: abs(self.predictors[x].profiled_accuracy - event.accuracy))
                if len(closest_acc_candidates) == 0:
                    self.simulator.bump_failed_request_stats(event)
                    return
                predictor = self.predictors[random.choice(closest_acc_candidates)]

            self.log.debug(f'enqueuing request at predictor {predictor}')
            predictor.enqueue_request(event, clock)
            self.assigned_requests[event.id] = predictor
            return
            
        else:
            raise ExecutorException(f'Unrecognized task assignment/job scheduling '
                                    f'algorithm for batch request processing')

    
    def apply_routing_table(self, routing_table={}):
        ''' Applies a given routing table to change the executor's current
        canary routing table
        '''
        self.log.debug(f'Executor: {self.isi}, Previous routing table: {self.canary_routing_table},'
                      f' new routing table: {routing_table}')

        self.canary_routing_table = routing_table
        return

    
    def get_largest_batch_size(self, model_variant, acc_type):
        largest_batch_sizes = self.simulator.get_largest_batch_sizes()

        if acc_type == 1:
            acc_type = 'CPU'
        elif acc_type == 2:
            acc_type = 'GPU_AMPERE'
        elif acc_type == 3:
            acc_type = 'VPU'
        elif acc_type == 4:
            acc_type = 'GPU_PASCAL'
        
        largest_batch_size = largest_batch_sizes[(acc_type, model_variant)]
        return largest_batch_size

    
    def predictors_by_variant_name(self):
        ''' Returns the list of predictors currently hosted by the executor,
        listing the variant name for each predictor
        '''
        predictor_dict = list(map(lambda x: x[1].variant_name, self.predictors.items()))
        return predictor_dict
                    