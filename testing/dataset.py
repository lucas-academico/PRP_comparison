# Define a function to get the dataset
import random
#import numpy as np
from itertools import combinations, permutations, product
from math import sqrt, ceil
from extra_functions_VNS import compute_distance_matrix, compute_distance_matrix2, vns_mvrp
from extracting_data import read_data


def get_dataset(nnodes, id_instance):
    random.seed(42)
    #extracted data
    
    Qv_init, sup_data, ret_data = read_data(nnodes,id_instance,1) #size, instance

    
    # Sets
    N = [f'Node{i}' for i in range(7)]
    F = [f'Family{i+1}' for i in range(1)]
    P = [f'Product{i+1}' for i in range(3)]
    T = list(range(1,6))  # Time periods
    V = [f'Vehicle{i+1}' for i in range(6)]
    K = [f'K{i+1}' for i in range(3)]
    end_nodes = N[1:]
    
    #parameters
    
    num_veh_per_type = {}
    for i in K:
        num_veh_per_type[i] = 2

    V_K = {}
    for i in V[:2]:
        V_K[i] = 'K1'
    for i in V[2:4]:
        V_K[i] = 'K2'
    for i in V[4:6]:
        V_K[i] = 'K3'
        
    vehicles_per_K = {k:0 for k in K}

    for v, k in V_K.items():
        vehicles_per_K[k] += 1

    
    Qv ={}
    Qv['K1'] = int(Qv_init*.1)
    Qv['K2'] = int(Qv_init*.2)
    Qv['K3'] = int(Qv_init*.3)
    
    
    locations = {}
    locations['Node0'] = (float(sup_data.loc[0,"X"]), float(sup_data.loc[0,"Y"]))
    locations['Node999'] = (float(sup_data.loc[0,"X"]), float(sup_data.loc[0,"Y"]))
    for i in N[1:]:
        ident = int(i[4:])
        locations[i] = (float(ret_data.loc[ident-1, "X"]), float(ret_data.loc[ident-1, "Y"]))
   
    distance_matrix = compute_distance_matrix(N, locations)
    distance_matrix2= compute_distance_matrix2(N, locations)
            
    P_f = {}
    if len(P)%(len(F))==0:
        a,b,c = 0, len(P)//(len(F)), len(P)//(len(F))
    else:
        a,b,c = 0, ceil(len(P)/len(F)), ceil(len(P)/len(F))
    
    for f in F:
        P_f[f] = {p: 1 if p in P[a:b] else 0 for p in P}
        a += c           
        b += c
    

    # Parameters
    #h = {(j,i): 1.2 if i != 'Node0' else 0.8 for j in P for i in N}
    #d = generate_demand(P, N[1:], T)
    d = {}
    h = {}
    for p in P:
        h[p,'Node0'] = float(sup_data.loc[0,"h"])
    
    for i in N[1:]:
        ident = int(i[4:])
        for p in P:
            h[p,i] = ceil(random.random() *  float(ret_data.loc[ident-1,"h"]))
            for t in T:
                d[p,i,t] = ceil(random.random() * float(ret_data.loc[ident-1,"DEM"])*2)
                
                
    #Qv = 500
    
    #Vol = {i: random.randint(1, 5) for i in P}  # Product volume
    Vol = {i: 1 for i in P}  # Product volume
    
    # Route creation ------------------------------------------------------------------------
    # locations = {}
    # for i in N[1:]:
    #     locations[i] = (random.randint(0, 100), random.randint(0, 100))
    
    # locations['Node0'] = (0, 0)  # Depot (Node0) at (0, 0)

    # distance_matrix = compute_distance_matrix(N, locations)
    
    #vol_of_dem = {}
    
    # for t in T:
    #     vol_of_dem[t] = {}
    #     for div in range(1,5):
    #         vol_of_dem[t][div] = {}
    #         for n in N[1:]:
    #             #vol_of_dem[t][n] = sum(d[p,n,t_on] * Vol[p] for p in P for t_on in T[t-1:])
    #             vol_of_dem[t][div][n] = sum(d[p,n,t_on+1] * Vol[p] for p in P for t_on in T[t-1:len(T)-1])+sum(d[p,n,t] * Vol[p] for p in P)/div
    
    I_max = {i: float(ret_data.loc[int(i[4:])-1,"I_MAX"]) if i != 'Node0' else 100000000 for i in N} # Max inventory
    
    #iteraciones = 2
    
    # for it in range(1,iteraciones):
    #     vol_of_dem[it] = {}
    #     if it<=5:
    #         for n in N[1:]:
    #             if round((I_max[n]/iteraciones) * it) < 1:
    #                 vol_of_dem[it][n] = 1
    #             else:
    #                 vol_of_dem[it][n] = round((I_max[n]/(iteraciones-1)) * it)
    #     else:
    #         for n in N[1:]:
    #             vol_of_dem[it][n] = round((I_max[n]/(iteraciones-1)) * it)

                
    #for it in range(iteraciones, iteraciones*2):
    # for it in range(1, iteraciones):        
    #     vol_of_dem[it] = {}
    #     for n in N[1:]:
    #         val = random.randint(1,int(I_max[n]))
    
    #         vol_of_dem[it][n] = val
    
    
    I_init = {(p,i): float(ret_data.loc[int(i[4:])-1,"I_INIT"]) if i != 'Node0' else float(sup_data.loc[0,"I_INIT"]) for i in N for p in P} 
    
    # max_dem = {}
    
    # for n in N[1:]:
    #     max_dem[n] = 6 - I_init['Product1',n]/d['Product1',n,1]
        
    # el stock inicial se define como un múltiplo de la demanda diaria.
    # los modelos son a 6 días
    # ¿cuánto voy a entregarle al nodo n en el peíodo t? un múltiplo de la demanda diaria que va entre 0 y los días no cubiertos por el stock inicial (max_dem)
         
    # for it in range(1, iteraciones):        
    #     vol_of_dem[it] = {}
    #     for n in N[1:]:
    #         val = random.randint(1,int(max_dem[n]))
    #         vol_of_dem[it][n] = d['Product1',n,1] * val
            
    
    # A, ct, ordered_routes = vns_mvrp(vol_of_dem,N,Qv,Vol,P,V,distance_matrix,iteraciones,max_iterations=10)
    
    # Adjacency Parameter (A[r, i]) and Fixed transport cost
    #A, ct = generate_all_routes_2(N, Qv, Vol, P, distance_matrix)
    
    # A, ct = generate_all_routes(N, Qv, Vol, P, distance_matrix)
    
    # R = list(ct.keys())
    
    eta_subsets = generate_eta_subsets(end_nodes)
    
    # Parameters------------------------------------------------------------------------
    Qm = {f:1000000 for f in F}  # Production capacity (INFINITA)
        
    #I_max = {i: 2000 if i != 'Node0' else 10000 for i in N} # Max inventory
    #I_max = {i: float(ret_data.loc[int(i[4:])-1,"I_MAX"]) if i != 'Node0' else 100000000 for i in N} # Max inventory
    
    #I_min = {(p,i): 2 if i != 'Node0' else 10 for p in P for i in N} # Min inventory
    
    I_min = {(p,i): 0 for p in P for i in N} # Min inventory

    #I_init = {(p,i): 15 if i != 'Node0' else 50 for p in P for i in N}
    
    
    # Holding cost
    
    #h = {(j,i): 1.2 if i != 'Node0' else 0.8 for j in P for i in N}
    
    cvp = {i:float(sup_data.loc[0,"PRODUCTION_COST"])  for i in P}  # Variable production cost

    cfp = {f:float(sup_data.loc[0,"SET_UP_COST"]) for f in F}  # Fixed production cost


    # Return the dataset
    return {
        'N': N, 'P': P, 'T': T, 'V': V, 'K': K, 'V_K': V_K,
        'Qm': Qm, 'd': d, 'I_max': I_max, 'I_min': I_min,
        'Qv': Qv, 'Vol': Vol, 'h': h, 'cvp': cvp, 'cfp': cfp, 'I_init': I_init,
        'num_veh_per_type':num_veh_per_type,
        'F': F, 'P_f':P_f , 'dist': distance_matrix, 'dist2': distance_matrix2, #'ordered_routes': ordered_routes , 
        'eta_subsets': eta_subsets, 'vehicles_per_K' : vehicles_per_K
    }



def generate_demand(P, N, T):
    """
    Generate demand 
    """
    
    demand = {}
    for t in T:
        for p in P:
            for i in N:
                    demand[p,i,t] = random.randint(6 , 15)
    
    return demand

#---------------------------------------------------------------------


def generate_all_routes(nodes, Qv, Vol, P, distance_matrix):
    """
    Generate all possible routes for a given list of nodes.
    """
    routes = {}
    ct = {} 
    route_counter = 1
    n0 = ('Node0',)
    nodes=nodes[1:]
    # Generate all subsets of nodes (including individual nodes and combinations of nodes)
    for subset_size in range(1,len(nodes)+1):
        for subset in permutations(nodes, subset_size):
            
            subset_with_depot = n0 + subset + n0
            
            route_cost = sum(distance_matrix[subset_with_depot[i]][subset_with_depot[i+1]] for i in range(len(subset_with_depot)-1))
            
            route_name = f'Route{route_counter}'
            route_counter += 1
        
            routes[route_name] = {node: 1 if node in subset else 0 for node in nodes}
            ct[route_name] = round(route_cost, 1)

    
    unique_routes = {}
    
    for route, nodes in routes.items():
        node_set = tuple(sorted(nodes.items()))
        cost = ct[route]
        
        if node_set not in unique_routes or cost < unique_routes[node_set][1]:
            unique_routes[node_set] = (route, cost)

    filtered_routes = {route: routes[route] for node_set, (route, cost) in unique_routes.items()}
    filtered_costs = {route: cost for node_set, (route, cost) in unique_routes.items()}

    return filtered_routes, filtered_costs
 
    
 
# def generate_all_routes_2(nodes, Qv, Vol, P, distance_matrix):
    
#     demand = 12
    
#     routes, distances = insertion_algorithm_2(nodes, distance_matrix, demand, Qv, Vol, P)    
    
#     return routes, distances
    
    

# def calculate_route_distance(route, distance_matrix):
#     """Calculate total distance of a route."""
#     total_distance = 0
#     for i in range(len(route) - 1):
#         total_distance += distance_matrix[route[i]][route[i + 1]]
#     return total_distance


# def insertion_algorithm_2(nodes, distance_matrix, demand, Qv, Vol, P):
#     # Initialize unvisited nodes (exclude depot)
#     routes = {}
#     distances = {}
#     route_count = 1
#     demand = (12,) #8,9,10,11,12
#     for dem in demand:
#         unvisited = set(nodes[1:])
#         while unvisited:
#             # Start a new route from depot
#             current_route = ['Node0']
#             current_demand = 0
#             current_nodes = []
            
#             while unvisited:
#                 last_node = current_route[-1]
#                 # Find nearest unvisited node
#                 nearest_node = None
#                 min_distance = float('inf')
                
#                 for node in unvisited:
#                     if distance_matrix[last_node][node] < min_distance:
#                         min_distance = distance_matrix[last_node][node]
#                         nearest_node = node
                
#                 # Check if adding the nearest node exceeds capacity
#                 #if nearest_node and current_demand + demands[nearest_node] <= vehicle_capacity:
#                 new_demand=0
#                 for p in P:
#                     new_demand += dem*Vol[p]
                
#                 if nearest_node and current_demand + new_demand <= Qv:
#                     current_route.append(nearest_node)
#                     current_nodes.append(nearest_node)                
#                     current_demand += new_demand
#                     unvisited.remove(nearest_node)
#                 else:
#                     # Close the route by returning to depot
#                     current_route.append('Node0')
#                     route_dict = {node: 1 if node in current_nodes else 0 for node in nodes[1:]}
#                     routes[f'Route{route_count}'] = route_dict
#                     distances[f'Route{route_count}'] = calculate_route_distance(current_route, distance_matrix)
#                     route_count += 1
#                     break
            
#             # If no nodes were added (e.g., all remaining nodes exceed capacity), close the route
#     #        if len(current_route) == 1 and unvisited:
#         current_route.append('Node0')
#         #route_dict = {node: 1 for node in nodes[1:]}
#         route_dict = {node: 1 if node in current_nodes else 0 for node in nodes[1:]}
#         routes[f'Route{route_count}'] = route_dict
#         distances[f'Route{route_count}'] = calculate_route_distance(current_route, distance_matrix)
#         #route_count += 1
                
#     return routes, distances
    
#---------------------------------------------------------------------
def generate_eta_subsets(Nc):
    subsets = []
    for r in range(2, len(Nc)+1):
        subsets.extend(combinations(Nc, r))
    return subsets
