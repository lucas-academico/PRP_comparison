# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 09:56:31 2025

@author: BANGHO
"""

import pyomo.environ as pyo
import time
from new_new_stochastic_dataset import get_dataset
    
m = pyo.ConcreteModel()

data = get_dataset(15, 1)

# Sets------------------------------------------------------------------------------------------------------------
m.N0 = pyo.Set(initialize=[data['N'][0]]) # Manufacturing site
m.N = pyo.Set(initialize=data['N'][1:])  # End-Nodes
m.Nend = pyo.Set(initialize=['Node999'])
m.Nt = m.N0|m.N|m.Nend #All-nodes
m.P = pyo.Set(initialize=data['P'])  # Products
m.F = pyo.Set(initialize=data['F'])  # Families
m.T = pyo.Set(initialize=data['T'])  # Time periods
m.R = pyo.Set(initialize=data['R'])  # Routes
m.V = pyo.Set(initialize=data['V'])  # Vehicles

# Parameters------------------------------------------------------------------------------------------------------------
m.P_f = pyo.Param(m.F, m.P, initialize = lambda m, f, p: data['P_f'][f][p], within=pyo.Binary)
m.A = pyo.Param(m.R, m.N, initialize=lambda m, r, n: data['A'][r][n], within=pyo.Binary)  # Adjacency
m.Qm = pyo.Param(m.F, initialize=data['Qm'], within=pyo.NonNegativeReals)  # Production capacity
m.d = pyo.Param(m.P, m.N, m.T, initialize=data['d'], default=0, within=pyo.NonNegativeReals)  # Demand
m.I_max = pyo.Param(m.Nt, initialize=data['I_max'], within=pyo.NonNegativeReals)  # Max inventory
m.I_min = pyo.Param(m.P, m.Nt, initialize=data['I_min'], within=pyo.NonNegativeReals)  # Min inventory
m.I_init= pyo.Param(m.P, m.Nt, initialize=data['I_init'], within=pyo.NonNegativeReals) # Initial Inventory
#m.Qv = Param(m.V, initialize=data['Qv'], within=NonNegativeReals)  # Vehicle capacity
m.Qv = pyo.Param(initialize=2*data['Qv'], within=pyo.NonNegativeReals)
m.Vol = pyo.Param(m.P, initialize=data['Vol'], within=pyo.NonNegativeReals)  # Product volume
m.ct = pyo.Param(m.R, initialize=data['ct'], within=pyo.NonNegativeReals)  # Transport cost
m.h = pyo.Param(m.P, m.Nt, initialize=data['h'], within=pyo.NonNegativeReals)  # Holding cost
m.cvp = pyo.Param(m.P, initialize=data['cvp'], within=pyo.NonNegativeReals)  # Variable cost
m.cfp = pyo.Param(m.F, initialize=data['cfp'], within=pyo.NonNegativeReals)  # Fixed cost


#Bounds_function------------------------------------------------------------------------------------------------------------
def Qp_bounds(model, p, t):
    for f in data['F']:
        if data['P_f'][f][p]==1:
            return (0, data['Qm'][f])  
        else:
            continue

def Qd_bounds(model, p, i, t):
    return (0, model.Qv/data['Vol'][p]) 
    
def I_bound(model, p, i, t):
    return (data['I_min'][p,i],data['I_max'][i]/data['Vol'][p])

def Nv_bound(model,r,t):
    return (0, len(m.V))

def ordered_node_pairs(m):
    return [(i, j) for i in m.Nt for j in m.Nt if i < j]

def dif_nodes(m):
    return [(i, j) for i in m.Nt for j in m.Nt if i != j]

def f_bounds(m, i, j, t):
    return (0,m.Qv)

# Decision variables------------------------------------------------------------------------------------------------------------
m.x_p = pyo.Var(m.F, m.T, within=pyo.Binary)  # Production binary
m.p = pyo.Var(m.P, m.T, bounds=Qp_bounds, within=pyo.NonNegativeReals)  # Production quantity
m.I = pyo.Var(m.P, m.N0|m.N, m.T, bounds=I_bound, within=pyo.NonNegativeReals)  # Inventory level
m.q = pyo.Var(m.P, m.N, m.T, bounds=Qd_bounds, within=pyo.NonNegativeReals)  # Delivery quantity
m.x = pyo.Var(ordered_node_pairs, m.T, within=pyo.Binary)
m.z = pyo.Var(m.N, m.T, within=pyo.Binary)
m.f = pyo.Var(dif_nodes, m.T, bounds=f_bounds, within=pyo.NonNegativeReals)

#new parameters
m.dist = pyo.Param(ordered_node_pairs, initialize= lambda m, i, j: data['dist2'][i][j], within=pyo.NonNegativeIntegers)
# Setting previous values ------------------------------------------------------------------------------------------------------------}

# for f in m.F:
#     for t in m.T:
#         m.x_p[f,t].fix(modelo.x[f,t])
#         #m.x_p[f,t].value = modelo.x[f,t]

# for p in m.P:
#     for t in m.T:
#         #m.p[p,t].fix(modelo.p[p,t])
#         m.p[p,t].value = modelo.p[p,t]
        
# for p in m.P:
#     for i in m.N0|m.N:
#         for t in m.T:
#             m.I[p,i,t].value = modelo.I[p,i,t]
            
# for p in m.P:
#     for i in m.N:
#         for t in m.T:
#             m.q[p,i,t].value = modelo.q[p,i,t]
            
# for i in m.N:
#     for t in m.T:
#         if sum(modelo.q[p,i,t].value for p in m.P) > 0.1:
#             m.z[i,t].value = 1
#         else:
#             m.z[i,t].value = 0
            
# for t in m.T:
#     if len(transport_results[t])==0:
#         continue
#     else:
#         for key in transport_results[t]:
#             m.x[key[0],key[1],t].value = transport_results[t][key[0],key[1]].value
        
#Contraints------------------------------------------------------------------------------------------------------------    

def production_limit_rule(m, f, t):
    
    return sum(m.p[p, t] for p in m.P if m.P_f[f,p]) <= m.x_p[f, t] * m.Qm[f]

m.ProductionLimit = pyo.Constraint(m.F, m.T, rule=production_limit_rule)


# model created following Manousakis (2022)
def equation2(m, i, t):
    return sum(m.x[j,i,t] for j in m.Nt if i>j) + sum(m.x[i,j,t] for j in m.Nt if i<j) == 2 * m.z[i,t]

m.Eq2 = pyo.Constraint(m.N, m.T, rule=equation2)

def equation3(m, t):
    return sum(m.x['Node0',j,t] for j in m.N) <= len(m.V)

m.Eq3 = pyo.Constraint(m.T, rule=equation3)

def equation4(m, t):
    return sum(m.x['Node0',j,t] for j in m.N) == sum(m.x[i,'Node999',t] for i in m.N)

m.Eq4 = pyo.Constraint(m.T, rule=equation4)

def equation5(m, i, j, t):
    return m.f[i,j,t] + m.f[j,i,t] == m.Qv * m.x[i,j,t] 

m.Eq5 = pyo.Constraint(ordered_node_pairs, m.T, rule=equation5)

def equation6(m, i, t):
    return sum(m.f[i,j,t] for j in m.Nt if i!=j) == m.Qv * m.z[i,t] - sum(m.q[p, i, t] for p in m.P)

m.Eq6 = pyo.Constraint(m.N, m.T, rule=equation6)

def equation7(m, t):
    return sum(m.f['Node0',j,t] for j in m.N) == sum(m.q[p,i,t] for i in m.N for p in m.P)

m.Eq7 = pyo.Constraint(m.T, rule=equation7)
 
def equation8(m, t):
    return sum(m.f[i,'Node999',t] for i in m.N) == 0

m.Eq8 = pyo.Constraint(m.T, rule=equation8)    

#Valid inequalities----------------------------------

# def inequality1(m, i, t):
#     return m.x['Node0', i, t] <= m.z[i,t]

# m.Ineq1 = pyo.Constraint(m.N, m.T, rule=inequality1)

# def inequality2(m, i, j, t):
#     if i == 'Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:
#         return m.x[i, j, t] <= m.z[i,t]

# m.Ineq2 = pyo.Constraint(ordered_node_pairs, m.T, rule=inequality2)

# def inequality3(m, i, j, t):
#     if i == 'Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:
#         return m.x[i, j, t] <= m.z[j,t]

# m.Ineq3 = pyo.Constraint(ordered_node_pairs, m.T, rule=inequality3)

# def inequality4(m, p, i, j, t):
#     if t==1:
#         return pyo.Constraint.Skip
#     elif i=='Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:    
#         return m.f[p, i, j, t] >=  m.d[p, j, t] * m.x[i, j, t] - m.I[p,j,t-1]

# m.Ineq4 = pyo.Constraint(m.P, ordered_node_pairs, m.T, rule=inequality4)

# def inequality5(m, p, i, j, t):
#     if t==1:
#         return pyo.Constraint.Skip
#     elif i=='Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:
#         return m.f[p, j, i, t] >=  m.d[p, i, t] * m.x[i, j, t] - m.I[p,i,t-1]

# m.Ineq5 = pyo.Constraint(m.P, ordered_node_pairs, m.T, rule=inequality5)

# def inequality6(m, p, i, j, t):
#     if t==1:
#         return pyo.Constraint.Skip
#     elif i=='Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:
#         return m.f[p, i, j, t] <=  m.Qv * m.x[i, j, t]

# m.Ineq6 = pyo.Constraint(m.P, ordered_node_pairs, m.T, rule=inequality6)

# def inequality7(m, p, i, j, t):
#     if t==1:
#         return pyo.Constraint.Skip
#     elif i=='Node0':
#         return pyo.Constraint.Skip
#     elif j == 'Node999':
#         return pyo.Constraint.Skip
#     else:
#         return m.f[p, j, i, t] <=  m.Qv * m.x[i, j, t]

# m.Ineq7 = pyo.Constraint(m.P, ordered_node_pairs, m.T, rule=inequality7)


#Inventory at Node 0----------------------------------
def inventory_balance_manufacturing_rule(m, p, t):
    if t == 1:
        return (
            m.I[p,'Node0', t] ==
            m.I_init[p,'Node0'] +
            m.p[p,t] -         # CAMBIÉ, LA PRODUCCIÓN IMPACTA EN EL MISMO DÍA 
            sum(m.q[p, j, t] for j in m.N)
        )
    else:
        return (
            m.I[p, 'Node0', t] ==
            m.I[p, 'Node0', t-1] +
            m.p[p, t] -         # CAMBIÉ, LA PRODUCCIÓN IMPACTA EN EL MISMO DÍA 
            sum(m.q[p, i, t] for i in m.N)
        )
    
m.InventoryBalanceManufacturing = pyo.Constraint(m.P, m.T, rule=inventory_balance_manufacturing_rule)

#Inventory at end-Node ----------------------------------
def inventory_balance_end_nodes_rule(m, p, i, t):
    if t == 1:
        #return Constraint.Skip
        return (
            m.I[p, i, t] ==
            m.I_init[p, i] +
            m.q[p, i, t] - 
            m.d[p, i, t]
            )

    return (
        m.I[p, i, t] ==
        m.I[p, i, t-1] +
        m.q[p, i, t] - # for r in m.R if m.A[r, i])
        m.d[p, i, t]
        )

m.InventoryBalanceEndNodes = pyo.Constraint(m.P, m.N, m.T, rule=inventory_balance_end_nodes_rule)

def inventory_min_rule(m, p, i, t):
    return m.I_min[p, i] <= m.I[p, i, t]

m.MinInventoryLimits = pyo.Constraint(m.P, m.N0|m.N, m.T, rule=inventory_min_rule)

def inventory_max_rule(m, p, i, t):
    return m.I[p, i, t] * m.Vol[p] <= m.I_max[i]

m.MaxInventoryLimits = pyo.Constraint(m.P, m.N0|m.N, m.T, rule=inventory_max_rule)    

#FO ----------------------------------
def objective_function(m):
    return (
        sum(m.x_p[f, t] * m.cfp[f] for f in m.F for t in m.T) +
        sum(m.p[p, t] * m.cvp[p] for p in m.P for t in m.T) +
        sum(m.I[p, i, t] * m.h[p, i] for p in m.P for i in m.N0|m.N for t in m.T) +
        sum(m.x[i, j, t] * m.dist[i, j] for i in m.Nt for j in m.Nt if i<j for t in m.T)
        )

m.Objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Create the solver
opt = pyo.SolverFactory('gurobi')
#opt = SolverFactory('gurobi_persistent', solver_oi='python')
#opt.set_instance(m)
opt.options['TimeLimit'] = 50  # Time limit (in seconds)
opt.options['MIPGap'] = 0.0001    # Optimality gap tolerance
opt.options['FeasibilityTol'] = 0.01

# Solve the model
results = opt.solve(m,  warmstart = True, tee=True)  