import numpy as np
import random
import math
import copy
from multiprocessing import Pool
import time
import utils

def single_experiment(sa, start_time, time_out, turning_point, temp, 
                      anneal_schedule, trans, graph, ro, max_sample):
    cost_record = []
    itr = 0
    while True:
        if time.time() > time_out:
            break
        if turning_point != None:
            if time.time() - start_time > turning_point:
                trans = "t1"
        if itr % anneal_schedule == 0:
            temp = temp*((0.9)**(itr/anneal_schedule))
        sa.single_transition(ro = ro, max_sample = max_sample, temp = temp, trans = trans)
        time_cost = (time.time() - start_time, utils.cost(sa.current_state, graph))
        cost_record.append(time_cost)
        itr += 1
    return cost_record

def single_experiment_itr(sa, max_itr, temp, anneal_schedule, trans, graph, ro, max_sample):
    cost_record = []
    for itr in range(max_itr):
        if (itr+1) % anneal_schedule == 0:
            temp = temp*((0.9)**((itr+1)/anneal_schedule))
        sa.single_transition(ro = ro, max_sample = max_sample, temp = temp, trans = trans)
        cost_record.append(utils.cost(sa.current_state, graph))
    return cost_record

def extract(repeat_result):
    x_result = []
    y_result = []
    for i in range(len(repeat_result)):
        x = [t[0] for t in repeat_result[i]]
        y = [t[1] for t in repeat_result[i]]
        x_result.append(x)
        y_result.append(y)
    return x_result, y_result

def multiprocess_SA(repeat, sa, start_time, time_out, turning_point,
                    temp, anneal_schedule, trans, graph, ro, max_sample):
    param = [(sa, start_time, time_out, turning_point, temp, anneal_schedule, trans, graph, ro, max_sample) for i in range(repeat)]
    pool = Pool()
    repeat_result = pool.starmap(single_experiment, param)
    x_result, y_result = extract(repeat_result)
    converge_points = [sum(result_cost[-10:])/10 for result_cost in y_result]
    picked_idx = converge_points.index(min(converge_points))
    return x_result[picked_idx], y_result[picked_idx]

def multiprocess_for_itr(repeat, sa, max_itr, temp, anneal_schedule, trans, graph, ro, max_sample):
    param = [(sa, max_itr, temp, anneal_schedule, trans, graph, ro, max_sample) for i in range(repeat)]
    pool = Pool()
    repeat_result = pool.starmap(single_experiment_itr, param)
    converge_points = [sum(candidate_cost[-10:])/10 for candidate_cost in repeat_result]
    print(converge_points)
    picked_idx = converge_points.index(min(converge_points))
    x = [i for i in range(max_itr)]
    return x, repeat_result[picked_idx]