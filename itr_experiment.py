# %%
import random
import numpy as np
import utils
import anneal
import matplotlib.pyplot as plt
import pickle
# %%
############ Hyperparameters
vertices = 50
farthest = 10
neighbor = "reverse"
trans1 = "t1"
trans2 = "t2"
trans3 = "t3"
initial_temp = 1
ro = 0.9
max_sample = 20
max_iter = 1000
iter_for_t1 = 5000
use_existed_graph = True
############

# %%
filename = "v" + str(vertices) + " graph_instance.pickle"
if use_existed_graph == False:
    graph = anneal.construct_graph(vertices = vertices, farthest = farthest)
    graph.set_distance()
    # store the graph into a pickle file. We can use the same graph in itr_experiment    
    with open(filename, "wb") as f:
        pickle.dump(graph, f, -1)
else:
    graph = pickle.load(open(filename, "rb", -1))

# %%
time = [i for i in range(max_iter)]
time_for_t1 = [i for i in range(iter_for_t1)]
num_plots = 3
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
labels = []  # label for each curve in the plot
    
# %%
sa1 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans1)
cost1 = []
for itr in range(iter_for_t1):
    if itr % 100 == 0:
        temp = initial_temp*((0.9)**(itr/100))
        sa1.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    else:
        sa1.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    cost1.append(utils.cost(sa1.current_state, graph))
plt.plot(time_for_t1, cost1)
labels.append("T1")
print("Finished transition 1")

# %%    
sa2 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans2)
cost2 = []
for itr in range(max_iter):
    if itr % 100 == 0:
        temp = initial_temp*((0.9)**(itr/100))
        sa2.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    else:
        sa2.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    cost2.append(utils.cost(sa2.current_state, graph))
plt.plot(time, cost2)
labels.append("T2")
print("Finished transition 2")

# %%
sa3 = anneal.simulated_annealing(graph = graph, neighbor = neighbor, trans = trans3)
cost3 = []
for itr in range(max_iter):
    if itr % 100 == 0:
        temp = initial_temp*((0.9)**(itr/100))
        sa3.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    else:
        sa3.single_transition(ro = ro, max_sample = max_sample, temp = temp)
    cost3.append(utils.cost(sa3.current_state, graph))
plt.plot(time, cost3)
labels.append("T3")
print("Finished transition 3")

plt.legend(labels, ncol=3, loc='upper center',
           bbox_to_anchor=[0.5, 1.1],
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.show()