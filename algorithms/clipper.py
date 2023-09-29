import logging
from algorithms.base import SchedulingAlgorithm


class Clipper(SchedulingAlgorithm):
    def __init__(self, simulator, solution_file, logging_level=logging.INFO):
        SchedulingAlgorithm.__init__(self, 'Clipper')
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)

        self.simulator = simulator
        self.solution_file = solution_file
        self.solution_applied = False

        return

    def apply_solution(self):
        ''' Applies the solution from self.solution_file
        '''
        if self.solution_applied:
            return

        with open(self.solution_file, mode='r') as rf:
            lines = rf.readlines()
            required_predictors = eval(lines[1].rstrip('\n'))
            canary_dict = eval(lines[3].rstrip('\n'))
            ilp_x = eval(lines[5].rstrip('\n'))

            self.log.info(f'required_predictors: {required_predictors}')
            self.log.info(f'canary_dict: {canary_dict}')
            self.log.info(f'ilp_x: {ilp_x}')
            self.simulator.apply_ilp_solution(required_predictors, canary_dict, ilp_x)
            self.solution_applied = True

        return
