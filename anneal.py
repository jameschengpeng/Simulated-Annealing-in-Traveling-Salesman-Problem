import random
import numpy as np
import utils
import copy

# We assume that the graph is fully connected
# otherwise, we can simply set the distance between two non-adjacent vertices as +ve infinity
class construct_graph:
    # vertices indicate how many vertices are in this graph
    # farthest indicate the largest distance between any two vertices
    def __init__(self, vertices, farthest):
        self.vertices = vertices
        self.farthest = farthest

    def set_distance(self):
        self.dist_matrix = np.zeros(shape = (self.vertices, self.vertices), dtype = np.float)
        for i in range(self.vertices - 1):
            for j in range(i+1, self.vertices):
                dist = random.uniform(0, self.farthest)
                #dist = np.random.uniform(0,self.farthest,1)[0]
                self.dist_matrix[i][j] = self.dist_matrix[j][i] = dist

class simulated_annealing:
    # graph is an object of class construct_graph
    # neighbor can be "reverse" or "swap", the default value is "reverse"
    # transition can be of type 1, type 2, type 3 (refer to utils.py)
    def __init__(self, graph, neighbor = "reverse"):
        self.graph = graph
        self.neighbor = neighbor
        self.initial_state = list(np.random.permutation(graph.vertices))
        self.current_state = copy.deepcopy(self.initial_state)

    def single_transition(self, ro, max_sample, temp, trans = "t1"):
        if trans == "t1":
            self.current_state = utils.transition_1(self.graph, self.current_state,
                                                    temp, self.neighbor)
        elif trans == "t2":
            self.current_state = utils.transition_2(self.graph, self.current_state,
                                                    temp, self.neighbor, ro)
        elif trans == "t3":
            self.current_state = utils.transition_3(self.graph, self.current_state,
                                                    temp, self.neighbor, ro, max_sample)

