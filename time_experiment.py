# %%
import random
import numpy as np
import utils
import anneal
import matplotlib.pyplot as plt
import time
import pickle
import parallel_comp as pc
import os
# %%
############ Hyperparameters
vertices = 50
farthest = 10
neighbor = "swap"
trans1 = "t1"
trans2 = "t2"
trans3 = "t3"
ro = 0.9
max_sample = 10
max_running_time = 30
turning_point = 4 # in sa4, the running time for t2 before changing to t1
repeat = 4
############
# %%
num_plots = 3
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
labels = []  # label for each curve in the plot

filename = "v" + str(vertices) + " graph_instance.pickle"
if os.path.exists(filename):
    graph = pickle.load(open(filename, "rb", -1))
else:
    graph = anneal.construct_graph(vertices = vertices, farthest = farthest)
    graph.set_distance()
    # store the graph into a pickle file. We can use the same graph in itr_experiment    
    with open(filename, "wb") as f:
        pickle.dump(graph, f, -1)

# %%
sa1 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
start_time1 = time.time()
time_out1 = time.time() + max_running_time
temp1 = 1

x1, y1 = pc.multiprocess_SA(repeat = repeat, sa = sa1, start_time = start_time1, time_out = time_out1, 
                           turning_point = None, temp = temp1, anneal_schedule = 1000, 
                           trans = trans1, graph = graph, ro = ro, max_sample = max_sample)

plt.plot(x1, y1)
labels.append("T1")
print("Finished transition 1. The number of iteration is: " + str(len(y1)))
print("The sequence converges to " + str( sum(y1[-10:])/10 ))
converge_time1 = utils.achieve_converge(x1, y1)
print("Achieved convergent at " + str(converge_time1) + "s" + "\n")

# %%
sa2 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
start_time2 = time.time()
time_out2 = time.time() + max_running_time
temp2 = 0.6
x2, y2 = pc.multiprocess_SA(repeat = repeat, sa = sa2, start_time = start_time2, time_out = time_out2, 
                           turning_point = None, temp = temp2, anneal_schedule = 80, 
                           trans = trans2, graph = graph, ro = ro, max_sample = max_sample)
plt.plot(x2, y2)
labels.append("T2")
print("Finished transition 2. The number of iteration is: " + str(len(y2)))
print("The sequence converges to " + str( sum(y2[-10:])/10 ))
converge_time2 = utils.achieve_converge(x2, y2)
print("Achieved convergent at " + str(converge_time2) + "s" + "\n")

# %%
sa3 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
start_time3 = time.time()
time_out3 = time.time() + max_running_time
temp3 = 1
x3, y3 = pc.multiprocess_SA(repeat = repeat, sa = sa3, start_time = start_time3, time_out = time_out3, 
                           turning_point = None, temp = temp3, anneal_schedule = 30, 
                           trans = trans3, graph = graph, ro = ro, max_sample = max_sample)
plt.plot(x3, y3)
labels.append("T3")
print("Finished transition 3. The number of iteration is: " + str(len(y3)))
print("The sequence converges to " + str( sum(y3[-10:])/10 ))
converge_time3 = utils.achieve_converge(x3, y3)
print("Achieved convergent at " + str(converge_time3) + "s" + "\n")

# %%
# This is the hybrid of t2 and t1
# t1 and t2 cost approximately the same time to converge but t2 converges faster at first
# so at first we use t2, then change to t1

# sa4 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
# start_time4 = time.time()
# time_out4 = time.time() + max_running_time
# temp4 = 3
# x4, y4 = pc.multiprocess_SA(repeat = repeat, sa = sa4, start_time = start_time4, time_out = time_out4, 
#                            turning_point = turning_point, temp = temp4, anneal_schedule = 100, 
#                            trans = trans2, graph = graph, ro = ro, max_sample = max_sample)
# plt.plot(x4, y4)
# labels.append("T4")
# print("Finished transition 4. The number of iteration is: " + str(len(y4)))
# print("The sequence converges to " + str( sum(y4[-10:])/10 ))
# converge_time4 = utils.achieve_converge(x4, y4)
# print("Achieved convergent at " + str(converge_time4) + "s" + "\n")

# %%
plt.legend(labels, ncol=3, loc='upper center',
           bbox_to_anchor=[0.5, 1.1],
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
plt.xlabel("Running Time (seconds)")
plt.ylabel("Cost of the path")
plt.show()

