# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:22:37 2025

@author: BANGHO
"""

import pyomo.environ as pyo
import time
from dataset import get_dataset
from exporting import export_pyomo_variables_to_excel

import itertools 


m = pyo.ConcreteModel()
data = get_dataset(50, 1)

# Sets------------------------------------------------------------------------------------------------------------
m.N0 = pyo.Set(initialize=[data['N'][0]]) # Manufacturing site
m.N = pyo.Set(initialize=data['N'][1:])  # End-Nodes
m.Nt = m.N0|m.N #All-nodes
m.P = pyo.Set(initialize=data['P'])  # Products
m.F = pyo.Set(initialize=data['F'])  # Families
m.T = pyo.Set(initialize=data['T'])  # Time periods
m.K = pyo.Set(initialize=data['K'])  # Heterogeneous Fleet (type of vehicle)
#m.V = pyo.Set(initialize=data['V'])  # Vehicles
m.node_pairs = pyo.Set(within=m.Nt * m.Nt, initialize=[(i, j) for i in m.Nt for j in m.Nt if i != j])
m.V_K = pyo.Set(dimen=2, initialize=[(v, k) for v, k in data['V_K'].items()])


# Parameters------------------------------------------------------------------------------------------------------------
m.P_f = pyo.Param(m.F, m.P, initialize = lambda m, f, p: data['P_f'][f][p], within=pyo.Binary)
m.dist = pyo.Param(m.node_pairs, initialize = lambda m, i, j: data['dist2'][i][j], within=pyo.NonNegativeIntegers)
m.Qm = pyo.Param(m.F, initialize=data['Qm'], within=pyo.NonNegativeReals)  # Production capacity
m.d = pyo.Param(m.P, m.N, m.T, initialize=data['d'], default=0, within=pyo.NonNegativeReals)  # Demand
m.I_max = pyo.Param(m.Nt, initialize=data['I_max'], within=pyo.NonNegativeReals)  # Max inventory
m.I_min = pyo.Param(m.P, m.Nt, initialize=data['I_min'], within=pyo.NonNegativeReals)  # Min inventory
m.I_init= pyo.Param(m.P, m.Nt, initialize=data['I_init'], within=pyo.NonNegativeReals) # Initial Inventory
m.Qv = pyo.Param(m.K, initialize=data['Qv'], within=pyo.NonNegativeReals)
m.Vol = pyo.Param(m.P, initialize=data['Vol'], within=pyo.NonNegativeReals)  # Product volume
m.h = pyo.Param(m.P, m.Nt, initialize=data['h'], within=pyo.NonNegativeReals)  # Holding cost
m.cvp = pyo.Param(m.P, initialize=data['cvp'], within=pyo.NonNegativeReals)  # Variable cost
m.cfp = pyo.Param(m.F, initialize=data['cfp'], within=pyo.NonNegativeReals)  # Fixed cost

m.vehicles_per_K = pyo.Param(m.K, initialize=data["vehicles_per_K"], within=pyo.NonNegativeIntegers)

#Bounds_function------------------------------------------------------------------------------------------------------------
def Qp_bounds(model, p, t):
    for f in data['F']:
        if data['P_f'][f][p]==1:
            return (0, data['Qm'][f])  
        else:
            continue

def Qd_bounds(model, p, i, k, t):
    return (0, model.Qv[k])
    #return (0, 10000)

def I_bound(model, p, i, t):
    return (data['I_min'][p,i],data['I_max'][i]/data['Vol'][p])

def dif_nodes(m):
    return [(i, j) for i in m.Nt for j in m.Nt if i != j]

# Decision variables------------------------------------------------------------------------------------------------------------
m.x_p = pyo.Var(m.F, m.T, within=pyo.Binary)  # Production binary
m.p = pyo.Var(m.P, m.T, bounds=Qp_bounds, within=pyo.NonNegativeReals)  # Production quantity
m.I = pyo.Var(m.P, m.Nt, m.T, bounds=I_bound, within= pyo.NonNegativeReals)  # Inventory level
m.q = pyo.Var(m.P, m.N, m.K, m.T, bounds=Qd_bounds, within=pyo.NonNegativeReals)  # Delivery quantity
# m.e = Var(m.P, m.N, m.T, m.S, within=NonNegativeReals)  # Unmet demand
m.z_base = pyo.Var(m.N0, m.K, m.T, within = pyo.NonNegativeIntegers)
m.z = pyo.Var(m.N, m.K, m.T, within = pyo.Binary)
m.x = pyo.Var(m.node_pairs, m.K, m.T, within = pyo.Binary)

#Contraints------------------------------------------------------------------------------------------------------------    

def production_limit_rule(m, f, t):
    
    return sum(m.p[p, t] for p in m.P if m.P_f[f,p]) <= m.x_p[f, t] * min(m.Qm[f], sum(m.d[p, i, l] for p in m.P if m.P_f[f,p] for i in m.N for l in m.T if l>=t))

m.ProductionLimit = pyo.Constraint(m.F, m.T, rule=production_limit_rule)


#Inventory constraints --------------------------------------------------------------------
def inventory_balance_manufacturing_rule(m, p, t):
    if t == 1:
        return (
            m.I[p, 'Node0', t] ==
            m.I_init[p, 'Node0'] +
            m.p[p, t] -
            sum(m.q[p, j, k, t] for k in m.K for j in m.N)
        )
    else:
        return (
            m.I[p, 'Node0', t] ==
            m.I[p, 'Node0', t-1] +
            m.p[p, t] -
            sum(m.q[p, i, k, t] for k in m.K for i in m.N)
        )
    
m.InventoryBalanceManufacturing = pyo.Constraint(m.P, m.T, rule=inventory_balance_manufacturing_rule)


def inventory_balance_end_nodes_rule(m, p, i, t):
    if t == 1:  
        return (
            m.I[p, i, t] ==
            m.I_init[p, i] +
            sum(m.q[p, i, k, t] for k in m.K) - 
            m.d[p, i, t]
            )

    return (
        m.I[p, i, t] ==
        m.I[p, i, t-1] +
        sum(m.q[p, i, k, t] for k in m.K) -
        m.d[p, i, t]
        )

m.InventoryBalanceEndNodes = pyo.Constraint(m.P, m.N, m.T, rule=inventory_balance_end_nodes_rule)

def inventory_min_rule(m, p, i, t):
    return m.I_min[p, i] <= m.I[p, i, t]

m.MinInventoryLimits = pyo.Constraint(m.P, m.Nt, m.T, rule=inventory_min_rule)

def inventory_max_rule(m, p, i, t):
    return m.I[p, i, t] * m.Vol[p] <= m.I_max[i]

m.MaxInventoryLimits = pyo.Constraint(m.P, m.Nt, m.T, rule=inventory_max_rule)    


#Routing constraints --------------------------------------------------------------------
#based on Ahmed 2023

#eq 2.7.1
def delivery_vehicle_rule1(m, k, t):
    
    return sum(m.q[p, i, k, t] * m.Vol[p] for p in m.P for i in m.N) <= m.Qv[k] * m.z_base['Node0',k,t]

m.DeliveryRouteUsage1 = pyo.Constraint(m.K, m.T, rule=delivery_vehicle_rule1)

# #eq 2.7.2
def delivery_vehicle_rule2(m, i, k, t):
    
    return sum(m.q[p, i, k, t] * m.Vol[p] for p in m.P) <= m.Qv[k] * m.z[i,k,t]

m.DeliveryRouteUsage2 = pyo.Constraint(m.N, m.K, m.T, rule=delivery_vehicle_rule2)


#eq 2.8
def edges_1_rule(m,i,k,t):
    if i!= 'Node0':
        return sum(m.x[i,j,k,t]+m.x[j,i,k,t] for j in m.Nt if i != j) == 2 * m.z[i,k,t]
    else:
        return sum(m.x[i,j,k,t]+m.x[j,i,k,t] for j in m.Nt if i != j) == 2 * m.z_base[i,k,t]

m.edges_1 = pyo.Constraint(m.Nt, m.K, m.T, rule=edges_1_rule)

def max_vehicles_rule(m,k,t):
    return m.z_base['Node0',k,t] <= m.vehicles_per_K[k]

m.max_vehicles = pyo.Constraint(m.K, m.T, rule=max_vehicles_rule)


# # #eq 1.8
# def max_visits_per_node_rule(m, i, t):
#     return sum(m.z[i,v,t] for v in m.V) <= 1

# m.max_visits_per_node = pyo.Constraint(m.N, m.T, rule=max_visits_per_node_rule)

# # #eq 1.9
# def visit_or_not_rule(m,v,k,i,t):
    
#     #return sum(m.q[p, i, v, t] * m.Vol[p] for p in m.P) <= min(m.Qv[k],m.I_max[i], sum(m.d[p, i, l] for p in m.P for l in m.T if l>=t)) * m.z[i,v,t]
#     return sum(m.q[p, i, v, t] * m.Vol[p] for p in m.P) <= m.Qv[k] * m.z[i,v,t]

# m.visit_or_not = pyo.Constraint(m.V_K, m.N, m.T, rule=visit_or_not_rule)


# def same_in_out_rule(m,i,v,t):
#     return sum(m.x[i, j, v, t] for j in m.Nt if i!=j) == sum(m.x[j, i, v, t] for j in m.Nt if i!=j)

# m.same_in_out_rule = pyo.Constraint(m.Nt,m.V,m.T,rule=same_in_out_rule)

# # #it was present in previous models but it's notin Ahmed 2023



# # subtour elimination constraint ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Indexing: eta_id ∈ index of eta_subsets, e ∈ eta
m.eta_index = pyo.RangeSet(0, len(data['eta_subsets']) - 1)
m.e_in_eta = pyo.Set(dimen=2, initialize=lambda m: [
    (idx, e) for idx, eta in enumerate(data['eta_subsets']) for e in eta
])                

def subtour_elimination_rule1(m, eta_id, e, k, t):
    eta = data['eta_subsets'][eta_id]
    lhs = m.Qv[k] * sum(m.x[i, j, k, t]
              for i in eta for j in eta if i!=j)
    
    rhs = sum(m.Qv[k] * m.z[i, k, t] - sum(m.q[p, i, k, t] for p in m.P) for i in eta)
    
    return lhs <= rhs                

m.SubtourElimination1 = pyo.Constraint(m.e_in_eta, m.K, m.T, rule=subtour_elimination_rule1)

def subtour_elimination_rule2(m, eta_id, e, k, t):
    eta = data['eta_subsets'][eta_id]
    lhs = sum(m.x[i, j, k, t]
              for i in eta for j in eta if i!=j)
    
    rhs = sum(m.z[i, k, t] for i in eta) - m.z[e, k, t]
    
    return lhs <= rhs                

m.SubtourElimination2 = pyo.Constraint(m.e_in_eta, m.K, m.T, rule=subtour_elimination_rule2)


#chatgpt implementation-------------------------------------------------

# def subsets(S):
#     for r in range(2, len(S)+1):
#         for comb in itertools.combinations(S, r):
#             yield comb

# subsets_S = list(subsets(list(m.N.data())))

# # Indexed set of subsets
# m.Subsets = pyo.Set(initialize=range(len(subsets_S)))

# # Map subset index → nodes
# subset_map = {k: subsets_S[k] for k in m.Subsets}

# # Constraint
# def subset_rule(m, s, k, t):
#     S = subset_map[s]
#     lhs = m.Qv[k] * sum(m.x[i, j, k, t]
#                    for (i,j) in m.node_pairs if i in S and j in S)
        
#     # Right-hand side: sum_{i in S}(Q*z[i,t] - q[i,t])
#     rhs = sum(m.Qv[k] * m.z[i,k,t] - sum(m.q[p,i,k,t] for p in m.P) for i in S)
#     return lhs <= rhs

# m.SubsetConstr = pyo.Constraint(m.Subsets, m.K, m.T,  rule=subset_rule)

# m.hom_fleet_breaking = pyo.ConstraintList()    
# def hom_fleet_breaking_rule(m, t):
#     for v in range(len(data['V'])-1):
#         m.hom_fleet_breaking.add(m.z['Node0',data['V'][v+1],t,s]<=m.z['Node0',data['V'][v],t,s])

# for s in m.S:
#     for t in m.T:
#         hom_fleet_breaking_rule(m,s,t)

# think about it, teniendo "tipos" de vehículos no es tan lineal
# tal vez lo mejor sea hacer un conjunto de vehículos por tipo, y que dsps haya
# otro gran set que sea "vehículos"

#FO ----------------------------------
def objective_function(m):
    return (
        sum(m.x_p[f, t] * m.cfp[f] for f in m.F for t in m.T) +
        sum(m.p[p, t] * m.cvp[p] for p in m.P for t in m.T) +
        sum(m.I[p, i, t] * m.h[p, i] for p in m.P for i in m.Nt for t in m.T) 
        + sum(m.x[i, j, k, t] * m.dist[i, j] for (i,j) in m.node_pairs for k in m.K for t in m.T)
        )

m.Objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Create the solver
opt = pyo.SolverFactory('gurobi')
opt.options['TimeLimit'] = 60  # Time limit (in seconds)
opt.options['MIPGap'] = 0.00001    # Optimality gap tolerance
opt.options['FeasibilityTol'] = 0.01

# Solve the model
results = opt.solve(m,  warmstart = True, tee=True)  

prod_cost = sum(m.x_p[f, t].value * m.cfp[f] for f in m.F for t in m.T) + sum(m.p[p, t].value * m.cvp[p] for p in m.P for t in m.T)
inv_cost = sum(m.I[p, i, t].value * m.h[p, i] for p in m.P for i in m.N0|m.N for t in m.T)
distr_cost = sum(m.x[i, j, k, t].value * m.dist[i, j] for (i,j) in m.node_pairs for k in m.K for t in m.T)
total_cost = prod_cost  + distr_cost + inv_cost

print("-----------------------------------")
print("-----------------------------------")
print(f"prod cost {prod_cost}")
print("-----------------------------------")
print("-----------------------------------")
print(f"inv cost {inv_cost}")
print("-----------------------------------")
print("-----------------------------------")
print(f"dist cost {distr_cost}")
print("-----------------------------------")
print("-----------------------------------")
print(f"total cost {total_cost}")


export_pyomo_variables_to_excel(m, filename='3index_output.xlsx')
