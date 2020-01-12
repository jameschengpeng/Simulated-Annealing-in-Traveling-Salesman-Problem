# %%
import random
import numpy as np
import utils
import anneal
import matplotlib.pyplot as plt
import pickle
import os
import copy
import parallel_comp as pc
# %%
############ Hyperparameters
vertices = 200
farthest = 40
neighbor = "swap"
trans1 = "t1"
trans2 = "t2"
trans3 = "t3"
ro = 0.9
max_sample = 20
repeat = 4
############

# %%
filename = "v" + str(vertices) + " graph_instance.pickle"
if os.path.exists(filename) == False:
    graph = anneal.construct_graph(vertices = vertices, farthest = farthest)
    graph.set_distance()
    # store the graph into a pickle file. We can use the same graph in itr_experiment    
    with open(filename, "wb") as f:
        pickle.dump(graph, f, -1)
else:
    graph = pickle.load(open(filename, "rb", -1))

# %%
num_plots = 3
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
labels = []  # label for each curve in the plot
    
# %%
sa1 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
y1 = []
temp1 = 5
anneal1 = 3500 # reduce temperature after every 100 iterations
iter_for_t1 = 140000
print(temp1)
print(anneal1)
print(iter_for_t1)
x1, y1 = pc.multiprocess_for_itr(repeat = repeat, sa = sa1, max_itr = iter_for_t1, temp = temp1, 
                                 anneal_schedule = anneal1, trans = trans1, graph = graph, ro = ro, max_sample = max_sample)
plt.plot(x1, y1)
labels.append("T1")
print("Finished transition 1")
print("The sequence converges to " + str( sum(y1[-10:])/10 ))
converge_itr1 = utils.achieve_converge(x1, y1)
print("Achieved convergent at " + str(converge_itr1) + " iteration" + "\n")

# %%    
sa2 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
y2 = []
temp2 = 2
anneal2 = 150
iter_for_t2 = 3000
print(temp2)
print(anneal2)
print(iter_for_t2)
x2, y2 = pc.multiprocess_for_itr(repeat = repeat, sa = sa2, max_itr = iter_for_t2, temp = temp2, 
                                 anneal_schedule = anneal2, trans = trans2, graph = graph, ro = ro, max_sample = max_sample)
x2_l = x1.copy()
y2_l = y2+[y2[-1] for i in range(iter_for_t1-iter_for_t2)]
plt.plot(x2, y2)
labels.append("T2")
print("Finished transition 2")
print("The sequence converges to " + str( sum(y2[-10:])/10 ))
converge_itr2 = utils.achieve_converge(x2, y2)
print("Achieved convergent at " + str(converge_itr2) + " iteration" + "\n")


# %%
sa3 = anneal.simulated_annealing(graph = graph, neighbor = neighbor)
y3 = []
temp3 = 1
anneal3 = 30
iter_for_t3 = 550
print(temp3)
print(anneal3)
print(iter_for_t3)
x3, y3 = pc.multiprocess_for_itr(repeat = repeat, sa = sa3, max_itr = iter_for_t3, temp = temp3, 
                                 anneal_schedule = anneal3, trans = trans3, graph = graph, ro = ro, max_sample = max_sample)
x3_l = x1.copy()
y3_l = y3 + [y3[-1] for i in range(iter_for_t1-iter_for_t3)]
plt.plot(x3, y3)
labels.append("T3")
print("Finished transition 3")
print("The sequence converges to " + str( sum(y3[-10:])/10 ))
converge_itr3 = utils.achieve_converge(x3, y3)
print("Achieved convergent at " + str(converge_itr3) + " iteration" + "\n")

# %%
num_plots = 3
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
labels = []  # label for each curve in the plot
plt.plot(x1, y1)
labels.append("T1")
plt.plot(x2_l, y2_l)
labels.append("T2")
plt.plot(x3_l, y3_l)
labels.append("T3")
plt.legend(labels, ncol=3, loc='upper center',
           bbox_to_anchor=[0.5, 1.1],
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.show()

# %%
print("Finished transition 1")
print("The sequence converges to " + str( sum(y1[-10:])/10 ))
converge_itr1 = utils.achieve_converge(x1, y1)
print("Achieved convergent at " + str(converge_itr1) + " iteration" + "\n")

print("Finished transition 2")
print("The sequence converges to " + str( sum(y2[-10:])/10 ))
converge_itr2 = utils.achieve_converge(x2, y2)
print("Achieved convergent at " + str(converge_itr2) + " iteration" + "\n")

print("Finished transition 3")
print("The sequence converges to " + str( sum(y3[-10:])/10 ))
converge_itr3 = utils.achieve_converge(x3, y3)
print("Achieved convergent at " + str(converge_itr3) + " iteration" + "\n")

# %%
