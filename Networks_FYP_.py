#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:19:41 2020

@author: joelmcfarlane

Networks Final Year Project 
"""
#%% - Imports and loading Network.
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import collections
import community
from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
import time
import infomap
import matplotlib.colors as colors
import pathlib

#sns.set()


col_list = ['b','g','r','c','m','y','gray','darkorange','limegreen'\
            ,'aquamarine','lightsteelblue','teal','lightsalmon','olive',\
            'pink','dodgerblue','mediumslateblue','firebrick']


# CHOOSE DATASET.

# Anonymized facebook data
# G_fb = nx.read_edgelist('facebook_combined.txt', create_using = nx.Graph()\
# , nodetype = int)

# Karate Club Data
G_fb = nx.karate_club_graph()

# Dolphin Social Network
# G_fb = nx.read_gml('dolphins//dolphins.gml')

#  #Politician Facebook pages network
# df = pd.read_csv('facebook_clean_data//government_edges.csv')
# Graphtype = nx.Graph()
# G_fb = nx.convert_matrix.from_pandas_edgelist(df, source = 'node_1'\
#                             ,target = 'node_2',create_using=Graphtype)

#  # Disease to side effect
#df = pd.read_csv('Se-DoDecagon_sidefx.csv')
#Graphtype = nx.Graph()
#G_fb = nx.convert_matrix.from_pandas_edgelist(df, source = 'Side Effect Name'\
#                             ,target = 'Disease Class',create_using=Graphtype)

# # Words dataset
#G_fb = nx.read_gml('adjnoun//adjnoun.gml')

# # Political Books
# G_fb = nx.read_gml('polbooks//polbooks.gml')

# #Political Blogs
#G_fb = nx.read_gml('polblogs//polblogs.gml') #DOESNT WORK.


  #Amazon customers who bought also bought
#G_fb = nx.read_edgelist('amazon0302.txt', create_using = nx.Graph()\
# , nodetype = int)
  
#  Condensed Matter Collaborations.
#G_fb = nx.read_gml('cond-mat-2003//cond-mat-2003.gml')  #DOESNT WORK.

# Astro collaborations
#G_fb = nx.read_gml('astro-ph//astro-ph.gml')
  
# Co-appearance of characters in Les Miserables
# G_fb = nx.read_gml('lesmis//lesmis.gml')
  
#   BBall
#df = pd.read_csv('bb_net.csv')
#Graphtype = nx.Graph()
#G_fb = nx.convert_matrix.from_pandas_edgelist(df, source = 'Person 1'\
#                             ,target = 'Person 2',edge_attr = 'Weight',\
#                             create_using=Graphtype)
#durations = [i['Weight'] for i in dict(G_fb.edges).values()]
#d = dict(G_fb.degree)
#nx.draw(G_fb, nodelist=d.keys(), node_size=[v * 100 for v in d.values()])


print(nx.info(G_fb))
#%% - Sort the graph into different communities using Spectral Methods (Newman)
start_time = time.time()



partition = partition(G_fb)
print("Took {} seconds to detect communities".\
      format( (time.time() - start_time)))
#

size = float(len(set(partition.values())))
pos = nx.spring_layout(G_fb)
count = 0.

start_time = time.time()


# - Draw the graph.
for com in set(partition.values()) :
    
    count = count + 1.

    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G_fb, pos, list_nodes, node_size = 20,
                                node_color = col_list[int(count-1)]) #str(count / size))

nx.draw_networkx_edges(G_fb,pos,\
                         alpha=0.5)
#
#nx.draw_networkx_edges(G_fb,pos,width = durations,\
#                         alpha=0.5) #for bballnet only
plt.show()

print('The modularity is {}'.format(community.modularity(partition, G_fb,)))

print("Took {} seconds to draw graph".format( (time.time() - start_time)))
#%% - Sort the graph into different communities using Louvain Greedy Methods.

start_time = time.time() 
#main()

partition = community.best_partition(G_fb)
print("Took {} seconds to detect communities".\
      format( (time.time() - start_time)))


size = float(len(set(partition.values())))
pos = nx.spring_layout(G_fb)
count = 0.



start_time = time.time()
#main()
# - Draw the graph.
for com in set(partition.values()) :
    
    
    count = count + 1.

    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G_fb, pos, list_nodes, node_size = 80,with_labels=True,
                                node_color = col_list[int(count-1)]) #str(count / size))


nx.draw_networkx_edges(G_fb, pos,with_labels = True, alpha=0.5)
plt.show()


print('The modularity is {}'.format(community.modularity(partition, G_fb,)))

print("Took {} seconds to draw graph".format( (time.time() - start_time)))

#%% - INFOMAP Community detection

def find_communities(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id.
    """
    start_time = time.time() 
    im = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for source, target in G.edges:
        im.add_link(source, target)

    print("Find communities with Infomap...")
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    communities = im.get_modules()
    nx.set_node_attributes(G, communities, 'community')
    print("Took {} seconds to find communities".format( (time.time() - start_time)))
    return communities
    
def draw_network(G):
    # position map
    pos = nx.spring_layout(G)
    # community ids
    communities = list(nx.get_node_attributes(G, 'community').values())
    num_communities = max(communities) + 1

    # color map from http://colorbrewer2.org/
    cmap_light = colors.ListedColormap(
        ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed', num_communities)
    cmap_dark = colors.ListedColormap(
        ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', num_communities)

    # edges
    nx.draw_networkx_edges(G, pos)

    # nodes
    node_collection = nx.draw_networkx_nodes(
        G, pos=pos, node_color=communities, cmap=cmap_light)

    # set node border color to the darker shade
    dark_colors = [cmap_dark(v) for v in communities]
    node_collection.set_edgecolor(dark_colors)

    # Print node labels separately instead
    for n in G.nodes:
        plt.annotate(n,
                     xy=pos[n],
                     textcoords='offset points',
                     horizontalalignment='center',
                     verticalalignment='center',
                     xytext=[0, 2],
                     color=cmap_dark(communities[n]))

    # plt.axis('off')
    # pathlib.Path("output").mkdir(exist_ok=True)
    # print("Writing network figure to output/karate.png")
    # plt.savefig("output/karate.png")
    plt.show()

partition = find_communities(G_fb)
draw_network(G_fb)
#%% - Calculate degree distribution



G = G_fb
#G = nx.gnp_random_graph(100, 0.02)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

#plt.title("Degree Histogram")
# plt.ylabel("Count")
# plt.xlabel("Degree")

plt.xlabel('Count',fontsize = 20)
plt.ylabel('Degree',fontsize = 20)
plt.tick_params(axis='both',labelsize=15)
plt.legend(loc='best',prop={'size':15})


ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

plt.show()
#%%
# =============================================================================
# 
# =============================================================================
#%% -  COMPARE COMMUNITY STRUCTURE (KARATE)

actual = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:2,10:1,11:1,12:1,13:1,14:2,\
          15:2,16:1,17:1,18:2,19:1,20:2,21:1,22:2,23:2,24:2,25:2,26:2,27:2,\
          28:2,29:2,30:2,31:2,32:2,33:2}


# In this louvain method iteration, groups 0 and 1 correspond to Mr Hi's group,
# groups 2 and 3 correspond to Officers group. (Hi = 1, Officer = 2)
right = []
wrong = []

# Note adjust these labels depending on the specific community
for i in range(len(actual)):
    
    if partition[i] == 2 or partition[i] == 3:
        group = 1
    else:
        group = 2
    
    if group == actual[i]:
        right.append(i)
    else:
        wrong.append(i)
        
print('The success rate is {}'.format(len(right) / (len(right)+len(wrong))))
#%%
# count = 0
# Draw correct graph:
for com in set(actual.values()) :
    
    
    count = count + 1.

    list_nodes = [nodes for nodes in actual.keys()
                                if actual[nodes] == com]
    nx.draw_networkx_nodes(G_fb, pos, list_nodes, node_size = 250,with_labels=True,
                                node_color = col_list[int(count-1)]) #str(count / size))


nx.draw_networkx_edges(G_fb, pos,with_labels = True, alpha=0.5)
plt.show()
