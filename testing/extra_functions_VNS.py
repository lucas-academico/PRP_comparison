# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:02:19 2025

@author: BANGHO
"""

import numpy as np
import random
from math import sqrt

# Compute Euclidean distance matrix
def compute_distance_matrix(N, locations):
    #n = len(locations)
    dist = {node: {} for node in N}
    for i in range(len(N)):
        for j in range(i,len(N)):
            if i!=j:
                distance = int(sqrt((locations[N[i]][0] - locations[N[j]][0])**2 + 
                                 (locations[N[i]][1] - locations[N[j]][1])**2)+.5)
                dist[N[i]][N[j]] = distance
                dist[N[j]][N[i]] = distance
    return dist

def compute_distance_matrix2(N, locations):
    #n = len(locations)
    N2=N.copy()
    N2.append('Node999')
    dist = {node: {} for node in N2}
    for i in range(len(N2)):
        for j in range(i,len(N2)):
            if i!=j:
                distance = int(sqrt((locations[N2[i]][0] - locations[N2[j]][0])**2 + 
                                 (locations[N2[i]][1] - locations[N2[j]][1])**2)+.5)
                dist[N2[i]][N2[j]] = distance
                dist[N2[j]][N2[i]] = distance
    return dist


# Evaluate total distance of a route
def evaluate_route(route, distance_matrix):
    if not route:  # Empty route
        return 0.0
    # Distance: depot -> first -> ... -> last -> depot
    dist = 0  # Depot (Node0) to first
    for i in range(len(route)-1):
        dist += distance_matrix[route[i]][route[i + 1]]  # Between nodes
        
    return round(dist,1)

# Check if a solution is feasible
def is_feasible(routes, Qv, N, demands):
    # Check capacity
    for v, route in enumerate(routes):
        total_demand = sum(demands[node] for node in route[1:-1])
        if total_demand > Qv:
            return False
    # Check each customer (Node1 to Node14) is visited exactly once
    visited = set()
    for route in routes:
        visited.update(route)
    #visited.update(['Node0'])
    return visited == set(N)  # Node1 to Node14 visited exactly once

def construct_initial_solution(demands,N,Qv,Vol,distance_matrix):
    routes = []
    #distances = {}
    route_count = 0
    unvisited = set(N[1:])
    while unvisited:
        # Start a new route from depot
        current_route = ['Node0']
        # RANDOM START 
        next_node = random.choice(list(unvisited))
        current_route = ['Node0'] + [next_node]
        unvisited.remove(next_node)
        
        current_demand = demands[next_node]
            
        while unvisited:
            last_node = current_route[-1]
            # Find nearest unvisited node
            nearest_node = None
            min_distance = float('inf')
                
            for node in unvisited:
                if distance_matrix[last_node][node] < min_distance:
                    min_distance = distance_matrix[last_node][node]
                    nearest_node = node
                
            # Check if adding the nearest node exceeds capacity
            #if nearest_node and current_demand + demands[nearest_node] <= vehicle_capacity:
                
            
            if nearest_node and current_demand + demands[nearest_node] <= Qv:
                current_route.append(nearest_node)            
                current_demand += demands[nearest_node]
                unvisited.remove(nearest_node)
            else:
                # Close the route by returning to depot
                current_route.append('Node0')
                routes.append(current_route)

                break
            
            # If no nodes were added (e.g., all remaining nodes exceed capacity), close the route
    #        if len(current_route) == 1 and unvisited:
    current_route.append('Node0')
    #route_dict = {node: 1 for node in nodes[1:]}
    routes.append(current_route)
    #route_dict = {node: 1 if node in current_nodes else 0 for node in N[1:]}
    #routes[f'Route{route_count}'] = route_dict
    #distances[f'Route{route_count}'] = calculate_route_distance(current_route, distance_matrix)
    #route_count += 1
                
    return routes

def shake(solution, neighborhood_func, intensity=1):
    new_solution = [route[:] for route in solution]
    for _ in range(intensity):
        new_solution = neighborhood_func(new_solution)
    return new_solution

# Neighborhood 1: Swap two nodes between routes
def swap_neighborhood(solution):
    new_solution = [route[:] for route in solution]
    v1, v2 = random.sample(range(len(solution)), 2)
    if not new_solution[v1] or not new_solution[v2]:
        return new_solution
    node1 = random.choice(new_solution[v1][1:-1])
    node2 = random.choice(new_solution[v2][1:-1])
    
    new_solution[v1].remove(node1)
    random_index_v1 = random.randint(1, len(new_solution[v1])-1)
    new_solution[v1].insert(random_index_v1, node2)
    
    new_solution[v2].remove(node2)
    random_index_v2 = random.randint(1, len(new_solution[v2])-1)
    new_solution[v2].insert(random_index_v2, node1)
    
    return new_solution

# Neighborhood 2: Relocate a node to another route
def relocate_neighborhood(solution):
    new_solution = [route[:] for route in solution]
    v1, v2 = random.sample(range(len(solution)), 2)
    if not new_solution[v1] or not new_solution[v2]:
        return new_solution
    if len(new_solution[v1])<4:
        if len(new_solution[v2])<4:
            return new_solution
    if len(new_solution[v1])<len(new_solution[v2]):
        node = random.choice(new_solution[v2][1:-1])
        new_solution[v2].remove(node)
        random_index = random.randint(1, len(new_solution[v1])-1)
        new_solution[v1].insert(random_index ,node)
    else:
        node = random.choice(new_solution[v1][1:-1])
        new_solution[v1].remove(node)
        random_index = random.randint(1, len(new_solution[v2])-1)
        new_solution[v2].insert(random_index ,node)
    
    
    return new_solution

# Neighborhood 3: 2-Opt within a route
def two_opt_neighborhood(solution):
    new_solution = [route[:] for route in solution]
    v = random.randint(0, len(new_solution) - 1)  # Pick a random route
    if len(new_solution[v]) < 4:  # Need at least 3 nodes for 2-opt
        return new_solution
    # Pick two positions in the route to reverse the segment between them
    i, j = sorted(random.sample(range(1,len(new_solution[v])-1), 2))
    if j - i < 2:  # Ensure a segment to reverse
        return new_solution
    new_solution[v][i:j] = reversed(new_solution[v][i:j])  # Reverse the segment
    return new_solution

# Neighborhood 4: Cross-Exchange between two routes
def cross_exchange_neighborhood(solution):
    new_solution = [route[:] for route in solution]
    v1, v2 = random.sample(range(len(new_solution)), 2)
    if len(new_solution[v1]) < 3 or len(new_solution[v2]) < 3:  # Need at least 2 nodes per route without depot
        return new_solution
    # Pick segments to exchange
    i1 = random.randint(1, len(new_solution[v1]) - 2)
    i2 = random.randint(i1 + 1, len(new_solution[v1])-1)
    j1 = random.randint(1, len(new_solution[v2]) - 2)
    j2 = random.randint(j1 + 1, len(new_solution[v2])-1)
    # Extract segments
    segment1 = new_solution[v1][i1:i2]
    segment2 = new_solution[v2][j1:j2]
    # Remove segments
    new_solution[v1] = new_solution[v1][:i1] + new_solution[v1][i2:]
    new_solution[v2] = new_solution[v2][:j1] + new_solution[v2][j2:]
    # Insert segments
    new_solution[v1] = new_solution[v1][:i1] + segment2 + new_solution[v1][i1:]
    new_solution[v2] = new_solution[v2][:j1] + segment1 + new_solution[v2][j1:]
    return new_solution

# Local search
def local_search(solution, neighborhood_func, distance_matrix, demands, vehicles, N, Qv):
    current_solution = [route[:] for route in solution]
    current_cost = sum(evaluate_route(route, distance_matrix) for route in current_solution)
    improved = True
    
    while improved:
        improved = False
        best_neighbor = current_solution
        best_cost = current_cost
        
        # Step 1: Inter-route moves (Swap, Relocate, etc.)
        for _ in range(30):  # Limit tries
            neighbor = neighborhood_func(current_solution)
            if is_feasible(neighbor, Qv, N, demands):
                cost = sum(evaluate_route(route, distance_matrix) for route in neighbor)
                if cost < best_cost:
                    best_neighbor = neighbor
                    best_cost = cost
                    improved = True
                    
        # Step 2: Intra-route 2-Opt for each route
        for v in range(len(best_neighbor)):
            route = best_neighbor[v][1:-1]
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_route.append('Node0')
                    new_route.insert(0,'Node0')
                    new_solution = [r[:] for r in best_neighbor]
                    new_solution[v] = new_route
                    new_cost = sum(evaluate_route(r, distance_matrix) for r in new_solution)
                    if new_cost < best_cost:
                        best_neighbor = new_solution
                        best_cost = new_cost
                        improved = True
                        
        current_solution = best_neighbor
        current_cost = best_cost
        
    return current_solution, current_cost

# Convert solution to desired output format
def format_routes(routes, N):
    output_routes = {}
    for v in range(len(routes)):
        route_name = f"Route{v+1}"
        route_dict = {}
        # Initialize all nodes (Node1 to Node14) as 0
        for node in range(1, len(N)):
            route_dict[f"Node{node}"] = 0
        # Set visited nodes to 1
        for node in routes[v][1:-1]:
            route_dict[node] = 1
        output_routes[route_name] = route_dict
    return output_routes

# Compute distances for output
def compute_route_distances(routes, distance_matrix):
    distances = {}
    for v in range(len(routes)):
        route_name = f"Route{v+1}"
        distance = evaluate_route(routes[v], distance_matrix)
        distances[route_name] = round(distance, 1)
    return distances

# Variable Neighboorhood Search algorithm
def vns_mvrp(all_demands,N,Qv,Vol,P,V,distance_matrix,iteraciones,max_iterations=20):
    # Initialize
    
    global_solution = []
    # Neighborhood structures
    #neighborhoods = [swap_neighborhood, relocate_neighborhood, two_opt_neighborhood, cross_exchange_neighborhood]
    neighborhoods = [swap_neighborhood, relocate_neighborhood, cross_exchange_neighborhood]
    k_max = len(neighborhoods)
    
    for t in range(1,iteraciones):
        #for div in range(1,5):
            #demands = all_demands[t][div]
            demands = all_demands[t]
            best_solution = construct_initial_solution(demands,N,Qv,Vol,distance_matrix)
            best_cost = sum(evaluate_route(route, distance_matrix) for route in best_solution)
            
            print("First best cost:", best_cost)
            
            original_route = []
            if len (best_solution) < 2:
                original_route = best_solution[0]
                length = len(original_route)                
                r1 = original_route[:int(length/2)] + ['Node0']
                r2 = ['Node0'] + original_route[int(length/2):]
                best_solution = [r1,r2]
            
            iteration = 1
            # VNS loop
            while iteration < max_iterations:
                k = 0
                while k < k_max:
                    current_solution = shake(best_solution, neighborhoods[k], intensity = iteration)
                    if not is_feasible(current_solution, Qv, N, demands):
                        k += 1
                        continue
                    current_solution, current_cost = local_search(current_solution, neighborhoods[k], 
                                                                 distance_matrix, demands, V, N, Qv)
                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost
                        k = 0
                    else:
                        k += 1
                iteration += 1
                
            print("Last best cost:", best_cost)
            
            for route in best_solution:
                global_solution.append(route)
                
            if len(original_route)>0:
                global_solution.append(original_route)
        
    
    # Convert each route to a tuple (since lists aren't hashable) and use a set to remove duplicates
    # Then convert back to a list, keeping the first occurrence
    unique_routes = list(dict.fromkeys(map(tuple, global_solution)))
    # Convert tuples back to lists for the final result
    unique_routes = [list(route) for route in unique_routes]
    
    # Format output
    routes_output = format_routes(unique_routes, N)
    ordered_routes = {}
    for v in range(len(unique_routes)):
        route_name = f"Route{v+1}"
        ordered_routes[route_name] = unique_routes[v]
    
    distances_output = compute_route_distances(unique_routes, distance_matrix)
    # Run the algorithm
    
    unique_routes_2 = {}
    
    for route, nodes in routes_output.items():
        node_set = tuple(sorted(nodes.items()))
        cost = distances_output[route]
        
        if node_set not in unique_routes_2 or cost < unique_routes_2[node_set][1]:
            unique_routes_2[node_set] = (route, cost)

    filtered_routes = {route: routes_output[route] for node_set, (route, cost) in unique_routes_2.items()}
    filtered_costs = {route: cost for node_set, (route, cost) in unique_routes_2.items()}


    
    return filtered_routes, filtered_costs, ordered_routes

