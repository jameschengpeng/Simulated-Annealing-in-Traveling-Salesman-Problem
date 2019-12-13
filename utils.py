import numpy as np
import random
import math
import copy
from multiprocessing import Pool
import time

# graph is of class construct_graph
def cost(perm, graph):
    total_cost = 0
    for i in range(len(perm)):
        if i != len(perm) - 1:
            start = perm[i]
            end = perm[i+1]
            total_cost += graph.dist_matrix[start][end]
        else:
            start = perm[i]
            end = perm[0]
            total_cost += graph.dist_matrix[start][end]
    return total_cost

def reverse_subsequence(perm, i, j):
    head = copy.deepcopy(perm[:i])
    intermediate = copy.deepcopy(perm[i:j+1])
    tail = copy.deepcopy(perm[(j+1):])
    intermediate.reverse()
    return head + intermediate + tail

def swap_positions(perm, i, j):
    list_copy = copy.deepcopy(perm)
    list_copy[i], list_copy[j] = list_copy[j], list_copy[i]
    return list_copy

def get_new_perm(perm, neighbor):
    pos_list = [i for i in range(len(perm))]
    selected_pos = random.sample(pos_list, 2)
    selected_pos.sort()
    i = selected_pos[0]
    j = selected_pos[1]
    if neighbor == "reverse":
        new_perm = reverse_subsequence(perm, i, j)
    elif neighbor == "swap":
        new_perm = swap_positions(perm, i, j)
    return new_perm

# generate a new perm having smaller or larger cost that currenct cost
def get_eligible_new_perm(perm, neighbor, s_or_l, old_cost, graph):
    # generate a new perm with smaller or equal cost
    if s_or_l == "s":
        # choose one from neighbors with equal or less costs or stay in current state
        new_perm = get_new_perm(perm, neighbor)
        counter = 0
        while cost(new_perm, graph) > old_cost:
            counter += 1
            if counter > 1000:
                return copy.deepcopy(perm)
                break
            new_perm = get_new_perm(perm, neighbor)
        return new_perm
    elif s_or_l == "l":
        # choose one from neighbors with larger costs
        new_perm = get_new_perm(perm, neighbor)
        counter = 0
        while cost(new_perm, graph) <= old_cost:
            counter += 1
            if counter > 1000:
                return copy.deepcopy(perm)
                break
            new_perm = get_new_perm(perm, neighbor)
        return new_perm        

# sample_cost is a list of cost for sample neighbors
# return a perm according to the probability distribution
def softmax_selection(sample_perms, graph):
    sample_cost = [cost(p, graph) for p in sample_perms]
    denominator = sum([math.exp(-c) for c in sample_cost])
    prob_dist = [(math.exp(-c) / denominator) for c in sample_cost]
    cdf = [sum(prob_dist[:(i+1)]) for i in range(len(prob_dist))]
    determinant = random.uniform(0,1)
    for i in range(len(cdf)):
        if i == 0 and determinant <= cdf[i]:
            return sample_perms[i]
        elif (i > 0 and determinant > cdf[i-1] and determinant <= cdf[i]):
            return sample_perms[i]

def transition_1(graph, perm, temp, neighbor):
    new_perm = get_new_perm(perm, neighbor)
    old_cost = cost(perm, graph)
    new_cost = cost(new_perm, graph)
    if old_cost >= new_cost:
        return new_perm
    else:
        switch_prob = math.exp((old_cost - new_cost)/temp)
        rand_num = random.uniform(0,1)
        if rand_num <= switch_prob:
            return new_perm
        else:
            return perm

# ro is the total probability given to those neighbors with smaller costs
def transition_2(graph, perm, temp, neighbor, ro):
    determinant = random.uniform(0,1)
    old_cost = cost(perm, graph)
    s_or_l = None
    if determinant <= ro:
        s_or_l = "s"
        new_perm = get_eligible_new_perm(perm, neighbor, s_or_l, old_cost, graph)
        return new_perm
    else:
        s_or_l = "l"
        new_perm = get_eligible_new_perm(perm, neighbor, s_or_l, old_cost, graph)
        new_cost = cost(new_perm, graph)
        switch_prob = math.exp((old_cost - new_cost)/temp)
        rand_num = random.uniform(0,1)
        if rand_num <= switch_prob:
            return new_perm
        else:
            return perm

# ro is the total probability given to those neighbors with smaller costs
# max_sample is the maximum number of neighbors generated to determine the next state
def transition_3(graph, perm, temp, neighbor, ro, max_sample):
    determinant = random.uniform(0,1)
    old_cost = cost(perm, graph)
    s_or_l = None
    if determinant <= ro:
        s_or_l = "s"
    else:
        s_or_l = "l"
    # generate max_sample sample neighbors which all have less or more costs than old_cost
    # use softmax to assign weights (i.e. the probability of choosing this neighbor as the next state)
    sample_perms = []
    while len(sample_perms) <= max_sample:
        sample_perms.append(get_eligible_new_perm(perm, neighbor, s_or_l, old_cost, graph))
    new_perm = softmax_selection(sample_perms, graph)
    if s_or_l == "s":
        return new_perm
    else:
        new_cost = cost(new_perm, graph)
        switch_prob = math.exp((old_cost - new_cost)/temp)
        rand_num = random.uniform(0,1)
        if rand_num <= switch_prob:
            return new_perm
        else:
            return perm

def achieve_converge(x_val, y_val):
    convergent = y_val[-1]
    threshold = int(convergent) + 1
    for i in range(len(y_val)):
        if y_val[i] <= threshold:
            check_subseq = y_val[i:i+100]
            check_result = all(y <= threshold for y in check_subseq)
            if check_result == True:
                return x_val[i]

