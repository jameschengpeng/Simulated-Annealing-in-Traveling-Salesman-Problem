# %%
import random
import numpy as np
import utils
import anneal
import matplotlib.pyplot as plt
import time
import pickle
# %%
############ Hyperparameters
vertices = 100
farthest = 10
neighbor = "reverse"
trans1 = "t1"
trans2 = "t2"
trans3 = "t3"
initial_temp = 1
ro = 0.9
max_sample = 20
max_running_time = 30
use_existed_graph = True
############
# %%
num_plots = 3
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
labels = []  # label for each curve in the plot

filename = "v" + str(vertices) + " graph_instance.pickle"
if use_existed_graph == False:
    graph = anneal.construct_graph(vertices = vertices, farthest = farthest)
    graph.set_distance()
    # store the graph into a pickle file. We can use the same graph in itr_experiment    
    with open(filename, "wb") as f:
        pickle.dump(graph, f, -1)
else:
    graph = pickle.load(open(filename, "rb", -1))
#print(graph.dist_matrix)
# %%
sa1 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans1)
cost1 = []
start_time1 = time.time()
time_out1 = time.time() + max_running_time
itr1 = 0
temp1 = initial_temp
while True:
    if time.time() > time_out1:
        break
    if itr1 % 300 == 0:
        temp1 = initial_temp*((0.9)**(itr1/300))
        sa1.single_transition(ro = ro, max_sample = max_sample, temp = temp1)
    else:
        sa1.single_transition(ro = ro, max_sample = max_sample, temp = temp1)
    time_cost = (time.time() - start_time1, utils.cost(sa1.current_state, graph))
    cost1.append(time_cost)  
    itr1 += 1      
x1 = [t[0] for t in cost1]
y1 = [t[1] for t in cost1]
plt.plot(x1, y1)
labels.append("T1")
print("Finished transition 1. The number of iteration is: " + str(itr1))

# %   
sa2 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans2)
cost2 = []
start_time2 = time.time()
time_out2 = time.time() + max_running_time
itr2 = 0
temp2 = initial_temp
while True:
    if time.time() > time_out2:
        break
    if itr2 % 100 == 0:
        temp2 = initial_temp*((0.9)**(itr2/100))
        sa2.single_transition(ro = ro, max_sample = max_sample, temp = temp2)
    else:
        sa2.single_transition(ro = ro, max_sample = max_sample, temp = temp2)
    time_cost = (time.time() - start_time2, utils.cost(sa2.current_state, graph))
    cost2.append(time_cost)
    itr2 += 1
    #print(time_out2 - time.time())
x2 = [t[0] for t in cost2]
y2 = [t[1] for t in cost2]
plt.plot(x2, y2)
labels.append("T2")
print("Finished transition 2. The number of iteration is: " + str(itr2))

# %
sa3 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans3)
cost3 = []
start_time3 = time.time()
time_out3 = time.time() + max_running_time
itr3 = 0
temp3 = initial_temp
while True:
    if time.time() > time_out3:
        break
    if itr3 % 100 == 0:
        temp3 = initial_temp*((0.9)**(itr3/100))
        sa3.single_transition(ro = ro, max_sample = max_sample, temp = temp3)
    else:
        sa3.single_transition(ro = ro, max_sample = max_sample, temp = temp3)
    time_cost = (time.time() - start_time3, utils.cost(sa3.current_state, graph))
    cost3.append(time_cost)
    itr3 += 1
x3 = [t[0] for t in cost3]
y3 = [t[1] for t in cost3]
plt.plot(x3, y3)
labels.append("T3")
print("Finished transition 3. The number of iteration is: " + str(itr3))

plt.legend(labels, ncol=3, loc='upper center',
           bbox_to_anchor=[0.5, 1.1],
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
plt.xlabel("Running Time (seconds)")
plt.ylabel("Cost of the path")
plt.show()


# %%
