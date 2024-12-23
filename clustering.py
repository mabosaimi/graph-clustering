import numpy as np
from typing import List, Tuple
import copy
import random
import time
import statistics

class Graph:
    def __init__(self, num_nodes: int, node_weights: List[int], edge_weights: List[List[int]]):
        self.num_nodes = num_nodes
        self.node_weights = np.array(node_weights)
        self.edge_weights = np.array(edge_weights)

class Cluster:
    def __init__(self, id: int, lower_bound: int, upper_bound: int):
        self.id = id
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.nodes = np.array([], dtype=int)
        self.total_weight = 0

    def can_add_node(self, node_weight: int) -> bool:
        return self.lower_bound <= self.total_weight + node_weight <= self.upper_bound

    def add_node(self, node: int, weight: int):
        self.nodes = np.append(self.nodes, node)
        self.total_weight += weight

    def remove_node(self, node: int, weight: int):
        self.nodes = self.nodes[self.nodes != node]
        self.total_weight -= weight

def greedy_heuristic(graph: Graph, num_clusters: int, cluster_bounds: List[Tuple[int, int]]) -> List[Cluster]:
    """
    Implements an improved greedy heuristic to create initial clusters.
    """
    if len(cluster_bounds) == 1:
        cluster_bounds = cluster_bounds * num_clusters
    elif len(cluster_bounds) != num_clusters:
        raise ValueError("The number of cluster bounds must be either 1 or equal to num_clusters")

    clusters = [Cluster(i, *bounds) for i, bounds in enumerate(cluster_bounds)]
    
    # Get all edges and sort them by weight in descending order
    edges = []
    for i in range(graph.num_nodes):
        for j in range(i+1, graph.num_nodes):
            if graph.edge_weights[i][j] > 0:
                edges.append((i, j, graph.edge_weights[i][j]))
    edges.sort(key=lambda x: x[2], reverse=True)
    
    assigned_nodes = set()

    # Assign nodes to clusters based on edge weights
    for edge in edges:
        node1, node2, _ = edge
        if node1 in assigned_nodes and node2 in assigned_nodes:
            continue

        for cluster in clusters:
            if (node1 not in assigned_nodes and cluster.can_add_node(graph.node_weights[node1])):
                cluster.add_node(node1, graph.node_weights[node1])
                assigned_nodes.add(node1)
            if (node2 not in assigned_nodes and cluster.can_add_node(graph.node_weights[node2])):
                cluster.add_node(node2, graph.node_weights[node2])
                assigned_nodes.add(node2)
            if node1 in assigned_nodes and node2 in assigned_nodes:
                break

    # Assign any remaining nodes to the least filled cluster
    unassigned_nodes = set(range(graph.num_nodes)) - assigned_nodes
    for node in unassigned_nodes:
        least_filled_cluster = min(clusters, key=lambda c: c.total_weight)
        least_filled_cluster.add_node(node, graph.node_weights[node])

    # Balance clusters to meet lower bounds
    balance_clusters(graph, clusters)

    return clusters

def balance_clusters(graph: Graph, clusters: List[Cluster]):
    """
    Balances clusters to meet lower bounds by transferring nodes between clusters.
    """
    for cluster in clusters:
        while cluster.total_weight < cluster.lower_bound:
            donor_cluster = max(clusters, key=lambda c: c.total_weight)
            if donor_cluster == cluster:
                break  # No other cluster can donate
            
            for node in donor_cluster.nodes:
                if (donor_cluster.total_weight - graph.node_weights[node] >= donor_cluster.lower_bound and
                    cluster.total_weight + graph.node_weights[node] <= cluster.upper_bound):
                    donor_cluster.remove_node(node, graph.node_weights[node])
                    cluster.add_node(node, graph.node_weights[node])
                    break
            else:
                break  # No suitable node found to transfer

def calculate_score(graph: Graph, clusters: List[Cluster]) -> float:
    """
    Calculates the total score of a clustering solution.
    """
    return sum(calculate_cluster_score(graph, cluster) for cluster in clusters)

def calculate_cluster_score(graph: Graph, cluster: Cluster) -> float:
    """
    Calculates the score of a single cluster.
    """
    if len(cluster.nodes) < 2:
        return 0
    cluster_edges = graph.edge_weights[np.ix_(cluster.nodes, cluster.nodes)]
    return np.sum(cluster_edges) / 2

def hill_climbing(graph: Graph, initial_clusters: List[Cluster], iterations: int = 100) -> List[Cluster]:
    """
    Implements a hill climbing algorithm to improve the initial clustering solution.
    """
    current_clusters = copy.deepcopy(initial_clusters)
    current_score = calculate_score(graph, current_clusters)

    for _ in range(iterations):
        best_improvement = 0
        best_move = None

        for i, source_cluster in enumerate(current_clusters):
            for node in source_cluster.nodes:
                if source_cluster.total_weight - graph.node_weights[node] < source_cluster.lower_bound:
                    continue
                    
                for j, target_cluster in enumerate(current_clusters):
                    if i == j or not target_cluster.can_add_node(graph.node_weights[node]):
                        continue

                    # Simulate the move
                    source_cluster.remove_node(node, graph.node_weights[node])
                    target_cluster.add_node(node, graph.node_weights[node])

                    new_score = calculate_score(graph, current_clusters)
                    improvement = new_score - current_score

                    # Undo the simulated move
                    target_cluster.remove_node(node, graph.node_weights[node])
                    source_cluster.add_node(node, graph.node_weights[node])

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (node, i, j)

        if best_move:
            node, source_idx, target_idx = best_move
            current_clusters[source_idx].remove_node(node, graph.node_weights[node])
            current_clusters[target_idx].add_node(node, graph.node_weights[node])
            current_score += best_improvement
        else:
            break


    return current_clusters

def iterated_local_search(graph: Graph, initial_clusters: List[Cluster], cluster_limits: List[Cluster], 
                          iterations: int = 10, hc_iterations: int = 100, perturbation_strength: int = 2):
    """
    Iterated Local Search with Hill Climbing and Perturbation to improve clustering solution.
    """
    # Start from the initial solution
    best_clusters = copy.deepcopy(initial_clusters)
    best_edge_weight = calculate_score(graph, best_clusters)

    current_clusters = best_clusters
    current_edge_weight = best_edge_weight

    for _ in range(iterations):
        # Local Search (Hill Climbing)
        current_clusters = hill_climbing(graph, current_clusters, iterations=hc_iterations)
        current_edge_weight = calculate_score(graph, current_clusters)

        # Check if current solution is better than the best found so far
        if current_edge_weight > best_edge_weight:
            best_clusters = copy.deepcopy(current_clusters)
            best_edge_weight = current_edge_weight

        # Perturbation: Randomly move nodes between clusters
        current_clusters = perturb_solution(graph, current_clusters, perturbation_strength)

        # Recalculate the edge weight after perturbation
        current_edge_weight = calculate_score(graph, current_clusters)

    return best_clusters

def perturb_solution(graph: Graph, clusters: List[Cluster], perturbation_strength: int):
    """
    Perturb the current solution by moving nodes between clusters randomly.
    """
    new_clusters = copy.deepcopy(clusters)

    # Flatten all nodes across clusters and shuffle them to randomly select for perturbation
    all_nodes = np.concatenate([cluster.nodes for cluster in new_clusters if len(cluster.nodes) > 0])
    np.random.shuffle(all_nodes)

    for _ in range(min(perturbation_strength, len(all_nodes))):
        node = all_nodes[_]
        node_weight = graph.node_weights[node]

        # Find the source cluster
        source_cluster = next(cluster for cluster in new_clusters if node in cluster.nodes)

        # Find potential target clusters
        target_clusters = [cluster for cluster in new_clusters if cluster.id != source_cluster.id and cluster.can_add_node(node_weight)]

        if not target_clusters:
            continue  # Skip if no valid target cluster found

        # Randomly select a target cluster
        target_cluster = random.choice(target_clusters)

        # Move the node from source to target
        source_cluster.remove_node(node, node_weight)
        target_cluster.add_node(node, node_weight)

    return new_clusters

def run_hill_climbing_analysis(graph: Graph, num_clusters: int, cluster_bounds: List[Tuple[int, int]], num_runs: int = 10):
    """
    Runs hill climbing multiple times and collects statistics.
    """
    all_scores = []
    initial_scores = []
    best_scores = []
    execution_times = []

    for _ in range(num_runs):
        initial_clusters = greedy_heuristic(graph, num_clusters, cluster_bounds)
        initial_score = calculate_score(graph, initial_clusters)
        initial_scores.append(initial_score)

        start_time = time.time()
        final_clusters = hill_climbing(graph, initial_clusters, iterations=10)
        end_time = time.time()

        all_scores.append(calculate_score(graph, final_clusters))
        best_scores.append(calculate_score(graph, final_clusters))
        execution_times.append(end_time - start_time)

    avg_initial_score = statistics.mean(initial_scores)
    avg_best_score = statistics.mean(best_scores)
    std_dev_best_score = statistics.stdev(best_scores)
    avg_execution_time = statistics.mean(execution_times)
    std_dev_execution_time = statistics.stdev(execution_times)

    return {
        "average_initial_score": avg_initial_score,
        "average_best_score": avg_best_score,
        "std_dev_best_score": std_dev_best_score,
        "average_execution_time": avg_execution_time,
        "std_dev_execution_time": std_dev_execution_time,
        "all_scores": all_scores
    }

def run_iterated_local_search_analysis(graph: Graph, num_clusters: int, cluster_bounds: List[Tuple[int, int]], 
                                       num_runs: int = 10, ils_iterations: int = 10, hc_iterations: int = 100, 
                                       perturbation_strength: int = 2):
    """
    Runs iterated local search multiple times and collects statistics.
    """
    all_scores = []
    initial_scores = []
    best_scores = []
    execution_times = []

    for _ in range(num_runs):
        initial_clusters = greedy_heuristic(graph, num_clusters, cluster_bounds)
        initial_score = calculate_score(graph, initial_clusters)
        initial_scores.append(initial_score)

        start_time = time.time()
        
        final_clusters = iterated_local_search(graph, initial_clusters, cluster_bounds, 
                                               iterations=ils_iterations, 
                                               hc_iterations=hc_iterations, 
                                               perturbation_strength=perturbation_strength)
        end_time = time.time()

        best_edge_weight = calculate_score(graph, final_clusters)

        all_scores.append(best_edge_weight)
        best_scores.append(best_edge_weight)
        execution_times.append(end_time - start_time)

    avg_initial_score = statistics.mean(initial_scores)
    avg_best_score = statistics.mean(best_scores)
    std_dev_best_score = statistics.stdev(best_scores) if len(best_scores) > 1 else 0
    avg_execution_time = statistics.mean(execution_times)
    std_dev_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

    return {
        "average_initial_score": avg_initial_score,
        "average_best_score": avg_best_score,
        "std_dev_best_score": std_dev_best_score,
        "average_execution_time": avg_execution_time,
        "std_dev_execution_time": std_dev_execution_time,
        "all_scores": all_scores
    }
