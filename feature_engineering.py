import numpy as np
import pandas as pd
import networkx as nx
import time
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.io import loadmat


def read_input_graph(ID):

    filename = "./small_graphs/" + str(ID) + "_fiber.mat"
    if os.path.exists(filename):
        data = loadmat(filename)
        matrix = data['fibergraph'].toarray()
        matrix = np.array(matrix)
        graph = nx.from_numpy_matrix(matrix)
        return graph

def compute_edge_betweenness(metadata):
    
    IDs = metadata['URSI']
    flag=0
    edge_betweenness_dicts = []
    LCS=[]
    for i in range(len(IDs)):
        tmp=[]
        graph = read_input_graph(IDs[i])
        d=nx.edge_betweenness_centrality(graph)
        edge_betweenness_dicts.append(d)
        if flag == 0:
            LCS=list(d.keys())
            flag = 1
        for i in d.keys():
            if i in LCS:
                tmp.append(i)
        LCS=tmp

    edge_betweenness_vectors = []
    for dictionary in edge_betweenness_dicts:
        edge_betweenness_vector = []
        for key in LCS:
            edge_betweenness_vector.append(dictionary[key])
        edge_betweenness_vectors.append(edge_betweenness_vector)

    return edge_betweenness_vectors

def to_feature_matrix(feature_vectors,metadata):

    feature_vectors = np.array(feature_vectors)
    feature_matrix = pd.DataFrame.from_records(feature_vectors)
    feature_matrix['Math Capability'] = metadata['Math Capability']

    return feature_matrix

def extract_feature_matrices(metadata, basepath):
    IDs = metadata['URSI']
    clustering_coefficient_vectors = []
    local_efficiency_vectors = []
    participation_coefficient_vectors = []
    for i in range(len(IDs)):
        graph = read_input_graph(IDs[i])
        
        clustering_coefficient_vectors.append(list(nx.clustering(graph).values()))    
        local_efficiency_vectors.append([nx.global_efficiency(graph.subgraph(graph[v])) for v in graph])
        participation_coefficient_vectors.append(list(nx.hits(graph)[0].values()))
    
    edge_betweenness_vectors = compute_edge_betweenness(metadata)

    clustering_coefficient_matrix = to_feature_matrix(clustering_coefficient_vectors, metadata)
    local_efficiency_matrix = to_feature_matrix(local_efficiency_vectors, metadata)
    participation_coefficient_matrix = to_feature_matrix(participation_coefficient_vectors, metadata)
    edge_betweenness_matrix = to_feature_matrix(edge_betweenness_vectors, metadata)

    clustering_coefficient_matrix.to_csv(basepath+'matrix_clustering_coefficient.csv', index=False)
    local_efficiency_matrix.to_csv(basepath+'matrix_local_efficiency.csv', index=False)
    participation_coefficient_matrix.to_csv(basepath+'matrix_participation_coefficient.csv', index=False)
    edge_betweenness_matrix.to_csv(basepath+'matrix_edge_betweenness.csv', index=False)

    return clustering_coefficient_matrix, local_efficiency_matrix, participation_coefficient_matrix , edge_betweenness_matrix

def find_differentiating_points(feature_matrix, basepath, label):
    test_df = feature_matrix.groupby('Math Capability').mean().values
    plt.plot(test_df[0], color='Red')
    plt.plot(test_df[1], color='Blue')
    plt.ylabel(label)
    fig_name = basepath + 'graph_' + label + '.png'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    filename = basepath + 'order_' + label + '.csv'
    significant_points = np.argsort(abs(test_df[0] - test_df[1]))[::-1]
    np.savetxt(filename, significant_points.astype(int), fmt ='%d', delimiter=',')

def generate_feature_matrix(basepath):

    print("Stage 1/2: Feature Engineering:")
    try:
        # Create target Directory
        os.mkdir(basepath)
    except FileExistsError:
        print("Stage 1/2: Feature Engineering results already present at directory",basepath)
        print("Stage 1/2: Type 'rerun' to recompute results")
        choice = input("Stage 1/2: Press Enter to skip: ")
        if choice == 'rerun':
            os.system("rm -rf "+basepath)
            os.mkdir(basepath)
        else:
            print("Stage 1/2: Skipping Stage 1...")
            return

    print("Stage 1/2: Generating feature matrices for clustering_coefficient, local_efficiency,"
                                             +" participation_coefficient and edge_betweenness ...")
    metadata = pd.read_csv('metadata.csv')
    ccf_mat, local_eff_mat, part_coeff_mat, edge_bwness_mat = extract_feature_matrices(metadata, basepath)

    #ccf_mat = pd.read_csv(basepath + 'matrix_clustering_coefficient.csv')
    #local_eff_mat = pd.read_csv(basepath + 'matrix_local_efficiency.csv')
    #part_coeff_mat = pd.read_csv(basepath + 'matrix_participation_coefficient.csv')
    #edge_bwness_mat = pd.read_csv(basepath + 'matrix_edge_betweenness.csv')
    
    print("Stage 1/2: Dumping feature engineering results to directory:", basepath, "...")
    find_differentiating_points(ccf_mat, basepath, 'clustering_coefficient')
    find_differentiating_points(local_eff_mat, basepath, 'local_efficiency')
    find_differentiating_points(part_coeff_mat, basepath, 'participation_coefficient')
    find_differentiating_points(edge_bwness_mat, basepath, 'edge_betweenness')
    
    feature_df = pd.concat([ccf_mat[[19, 32, 21, 18, 22]], local_eff_mat[[19, 32, 21, 18, 22]],
                            part_coeff_mat[[52]], edge_bwness_mat[[232, 159, 269, 270]]], axis=1)

    feature_df['Math Capability'] = ccf_mat['Math Capability']
    print("Stage 1/2: Dumping feature values of chosen nodes/edges to "+basepath+"Engineered_Features.csv")
    feature_df.to_csv(basepath+'Engineered_Features.csv', index=False)
    print("Stage 1/2: Complete")
