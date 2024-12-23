import numpy as np
from clustering import Graph, run_hill_climbing_analysis, run_iterated_local_search_analysis, calculate_score, greedy_heuristic
class GraphInstance:
    def __init__(self, num_nodes, num_clusters, cluster_bounds, node_weights, edges):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.cluster_bounds = cluster_bounds  
        self.node_weights = node_weights      
        self.edges = edges                   

def parse(file_path):
    with open(file_path, 'r') as f:

        first_line = f.readline().strip().split()

        num_nodes = int(first_line[0])  # Number of nodes
        num_clusters = int(first_line[1])  # Number of clusters
        

        cluster_bounds = np.array(first_line[2:2 + 2*num_clusters], dtype=int).reshape(num_clusters, 2)

        W_index = 2 + 2*num_clusters
        node_weights = np.array(first_line[W_index + 1:W_index + 1 + num_nodes], dtype=int)

        edges = np.loadtxt(f, dtype=float)  


    return GraphInstance(num_nodes, num_clusters, cluster_bounds, node_weights, edges)

def convert_to_graph(instance):
    num_nodes = instance.num_nodes
    node_weights = instance.node_weights

    edge_weights = np.zeros((num_nodes, num_nodes), dtype=float)
    

    for edge in instance.edges:
        node1, node2, weight = int(edge[0]), int(edge[1]), edge[2]
        edge_weights[node1, node2] = weight
        edge_weights[node2, node1] = weight  
    
    return Graph(num_nodes, node_weights, edge_weights)

def main():
    Sparse = parse('Sparse82.txt')
    Sparse_graph = convert_to_graph(Sparse)
    
    RanReal = parse('RanReal480.txt')
    RanReal_graph = convert_to_graph(RanReal)
    
    print("Sparse graph")
    print("Greedy Score:", calculate_score(Sparse_graph, greedy_heuristic(Sparse_graph, Sparse.num_clusters, Sparse.cluster_bounds)))
    analysis = run_hill_climbing_analysis(Sparse_graph, Sparse.num_clusters, Sparse.cluster_bounds)
    print("Hill Climbing Analysis:")
    print("Average initial score:", analysis["average_initial_score"])
    print("Average best score:", analysis["average_best_score"])
    print("Standard deviation of best scores:", analysis["std_dev_best_score"])
    print("Average execution time:", analysis["average_execution_time"])
    print("Standard deviation of execution time:", analysis["std_dev_execution_time"])
    print("Best score:", max(analysis["all_scores"]))
    
    print("-------------------------------------------------------")
    
    analysis = run_iterated_local_search_analysis(Sparse_graph, Sparse.num_clusters, Sparse.cluster_bounds)
    print("Iterated Local Search Analysis:")
    print("Average initial score:", analysis["average_initial_score"])
    print("Average best score:", analysis["average_best_score"])
    print("Standard deviation of best scores:", analysis["std_dev_best_score"])
    print("Average execution time:", analysis["average_execution_time"])
    print("Standard deviation of execution time:", analysis["std_dev_execution_time"])
    print("Best score:", max(analysis["all_scores"]))
    
    print("-------------------------------------------------------")
    
    print("RanReal graph")
    print("Greedy Score:", calculate_score(RanReal_graph, greedy_heuristic(RanReal_graph, RanReal.num_clusters, RanReal.cluster_bounds)))
    analysis = run_hill_climbing_analysis(RanReal_graph, RanReal.num_clusters, RanReal.cluster_bounds)
    print("Hill Climbing Analysis:")
    print("Average initial score:", analysis["average_initial_score"])
    print("Average best score:", analysis["average_best_score"])
    print("Standard deviation of best scores:", analysis["std_dev_best_score"])
    print("Average execution time:", analysis["average_execution_time"])
    print("Standard deviation of execution time:", analysis["std_dev_execution_time"])
    print("Best score:", max(analysis["all_scores"]))
    
    print("-------------------------------------------------------")
    
    analysis = run_iterated_local_search_analysis(RanReal_graph, RanReal.num_clusters, RanReal.cluster_bounds)
    print("Iterated Local Search Analysis:")
    print("Average initial score:", analysis["average_initial_score"])
    print("Average best score:", analysis["average_best_score"])
    print("Standard deviation of best scores:", analysis["std_dev_best_score"])
    print("Average execution time:", analysis["average_execution_time"])
    print("Standard deviation of execution time:", analysis["std_dev_execution_time"])
    print("Best score:", max(analysis["all_scores"]))

if __name__ == "__main__":
    main()


