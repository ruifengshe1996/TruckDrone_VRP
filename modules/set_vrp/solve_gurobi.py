# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:09:03 2022

@author: ruife
"""
import networkx as nx
import gurobipy as gp
import numpy as np
import heapq

from itertools import product
from classes.tdvrp import TDVRP
from collections import defaultdict


class GurobiSolve(object):
    def __init__(self, vrp:TDVRP, G:nx.DiGraph) -> None:
        self.vrp = vrp
        self.G = G

        print(' \n initializing gurobi model')
        self.model = gp.Model() 
        self.obj = None


        # some convinient mappins
        # node-dummy
        node_dummy_dict = dict()
        node_dummy_dict[vrp.depot] = vrp.depot
        for region_key,region in vrp.regions.items():
            for node in region.boundary:
                node_dummy_dict[node] = 'dummy_'+str(region_key)
        self.node_dummy_dict = node_dummy_dict

        # prepare proper node demand
        node_demand = dict()
        node_demand[vrp.depot] = 0
        for region_key,region in vrp.regions.items():
            demand = int(vrp.regions[region_key].num_demand / 3 )
            node_demand['dummy_'+str(region_key)] = demand
            for node in region.boundary:
                node_demand[node] = demand
        self.node_demand = node_demand

        return

    def _add_variables(self):

        vrp = self.vrp
        G = self.G
        m = self.model
        

        intra_dr_cost = vrp.intra_dr_cost
        num_trucks = vrp.num_trucks

        # ================== adding variables ===============================
        
        
        # add x
        x = dict()
        for k in range(num_trucks):
            edge_dict = {(k,edge[0],edge[1]):G.edges[(edge[0],edge[1])]['weight'] for edge in G.edges}
            
            x[k] = m.addVars(edge_dict.keys(), 
                            obj=edge_dict, 
                            vtype=gp.GRB.BINARY, 
                            name='x'+str(k))
            
        
        # add tau
        tau = dict()
        for k in range(num_trucks):
            
            edge_dict = dict()
            
            for region_key, cost_dict in intra_dr_cost.items():
                for u,v in product(cost_dict.index,cost_dict.columns):
                    edge_dict[(k,u,v)] = cost_dict.loc[u,v] 
                    
            tau[k] = m.addVars(edge_dict.keys(), 
                            obj=edge_dict, 
                            vtype=gp.GRB.BINARY, 
                            name='tau'+str(k))
        
        # adding u
        u = m.addVars(G.edges,
                    obj = 0.0,
                    vtype = gp.GRB.INTEGER,
                    name = 'u')
        
        m.update()
        variables = {'x':x,'u':u,'tau':tau}
        self.variables = variables

        return

    def _add_constriants(self):
        
        m = self.model
        x = self.variables['x']
        u = self.variables['u']
        tau = self.variables['tau']


        vrp = self.vrp
        G = self.G


        num_trucks = vrp.num_trucks
        truck_capacity = vrp.truck_capacity
        depot = vrp.depot

        # ========================== adding  constraints ==========================
        
        M = 1e3
        cst_dict = dict()
        
        # flow conservation
        csts = dict()
        for k in range(num_trucks):
            for i in G.nodes:
                cst_linexp = gp.LinExpr()
                for j in G.adj[i]:
                    # outward edges
                    cst_linexp += x[k][(k,i,j)]
                    # inward edges
                    cst_linexp -= x[k][(k,j,i)]
                csts[(k,i)] = m.addConstr(cst_linexp == 0)
        cst_dict['flow_conservation'] = csts
        
        # # single truck traverse for each edge
        csts = dict()
        for edge in G.edges:
            cst_linexp = gp.LinExpr()
            for k in range(num_trucks):
                cst_linexp += x[k][(k,edge[0],edge[1])]
            csts[edge] = m.addConstr(cst_linexp <= 1)
        cst_dict['single_traverse'] = csts

        # single dispatch for each truck
        csts = dict()
        for k in range(num_trucks):
            cst_linexp = gp.LinExpr()
            for j in G.adj[depot]:
                # outward edges
                cst_linexp += x[k][(k,depot,j)]
            csts[k] = m.addConstr(cst_linexp <= 1)
        cst_dict['single_dispatch'] = csts
        

        # subtour elimination
        csts0 = dict()
        csts1 = dict()
        for edge in G.edges:
            cst_linexp = gp.LinExpr()
            for k in range(num_trucks):
                cst_linexp += x[k][(k,edge[0],edge[1])]
            
            # upper bound
            cst_linexp_ub = u[edge] - len(G.edges) * cst_linexp
            csts[0] = m.addConstr(cst_linexp_ub <= 0)

            # lower bound
            cst_linexp_lb = cst_linexp - u[edge]
            csts1[edge] = m.addConstr(cst_linexp_lb <= 0)
        csts2 = dict()
        for i in G.nodes:
            if i == depot:
                continue
            cst_linexp = gp.LinExpr()
            for j in G.adj[i]:
                cst_linexp += u[(i,j)]
                cst_linexp -= u[(j,i)]
                for k in range(num_trucks):
                    cst_linexp -= x[k][(k,i,j)]
            csts2[i] = m.addConstr(cst_linexp == 0)
        
        cst_dict['subtour_elimination'] = [csts0,csts1,csts2]

        # capacity
        csts = dict()
        for k in range(num_trucks):
            cst_linexp = gp.LinExpr()
            for edge in G.edges:
                cst_linexp += x[k][(k,edge[0],edge[1])] * self.node_demand[edge[1]]
            cst_linexp -= truck_capacity[k]
            csts[k] = m.addConstr(cst_linexp <= 0)
        cst_dict['capacity'] = csts
 
        # cluster visiting rule
        csts = dict()
        for n in G.nodes:
            if isinstance(n,str): # dummy node
                cst_linexp = gp.LinExpr()
                for i in G.adj[n]:
                    for k in range(num_trucks):
                        cst_linexp += x[k][(k,i,n)]
                csts[n] = m.addConstr(cst_linexp == 1)
        cst_dict['cluster_single_visit'] = csts
        
        csts = dict()
        for k in range(num_trucks):
            for j in G.nodes:
                if not isinstance(j,str) and j != depot:
                    cst_linexp = gp.LinExpr()
                    for i in G.adj[j]:
                        if isinstance(i,str): 
                            # outward edge to the dummy nume
                            cst_linexp -= x[k][(k,j,i)]
                        else:
                            # inward edges except from the dummy node
                            cst_linexp += x[k][(k,i,j)]
                    csts[(j,k)] = m.addConstr(cst_linexp == 0)
        cst_dict['cluster_continuous_visit'] = csts
        
        # tau definition
        csts0 = dict()
        csts1 = dict()
        csts2 = dict()
        for k in range(num_trucks):
            for _,i,j in tau[k].keys():
                
                dummy_i = self.node_dummy_dict[i]
                dummy_j = self.node_dummy_dict[j]
                
                # i,j belongs to the same region
                if dummy_i == dummy_j:
                    dummy = dummy_i
                    csts0[(k,i,j)] = m.addConstr(tau[k][(k,i,j)] - x[k][(k,i,dummy)] <= 0)
                    csts1[(k,i,j)] = m.addConstr(tau[k][(k,i,j)] - x[k][(k,dummy,j)] <= 0)
                    csts2[(k,i,j)] = m.addConstr(x[k][(k,i,dummy)] + x[k][(k,dummy,j)] - 1 - tau[k][(k,i,j)] <= 0)
        cst_dict['tau_definition'] = [csts0,csts1,csts2]
        m.update()

        self.constraints = cst_dict

        return 

    def _set_objective(self):
        # ================== setting objective ===============================
        m = self.model
        G = self.G

        x = self.variables['x']
        
        
        obj_linexp = m.getObjective()
        
        # edge travel cost and cpp already included
        # add dispatch cost
        
        depot = self.vrp.depot
        num_trucks = self.vrp.num_trucks
        truck_cost = self.vrp.truck_cost

        for k in range(num_trucks):
            t = gp.LinExpr()
            for edge in G.edges:
                if edge[0] == depot:
                    t += x[k][(k,edge[0],edge[1])]
            obj_linexp += t * truck_cost[k]
        
        m.setObjective(obj_linexp,gp.GRB.MINIMIZE)
        m.update()
        return

    def _set_solution(self,solution:dict[list]):
        ''' assign a priori feasible path solutions
        @param solution = {truck_key: [depot,node,...,depot]}
        '''
        m = self.model
        G = self.G

        depot = self.vrp.depot

        x = self.variables['x']
        u = self.variables['u']
        tau = self.variables['tau']

        # pre-set all variables to 0
        [var.setAttr('Start',0.0) for var in m.getVars()]

        u_ = dict.fromkeys(u,0)

        for k, path_ in solution.items():
            
            # convert to a compatible depot-boundary-dummy-boundary-depot path
            path = []
            while path_:
                node = path_.pop(0)
                if path:
                    if self.node_dummy_dict[path[-1]] == self.node_dummy_dict[node]:
                        path.append(self.node_dummy_dict[node])
                path.append(node)

            for i in range(len(path)-1):
                start = path[i]
                end = path[i+1]

                if i==0:
                    # depot - boundary
                    u_[(start,end)] = 1
                else:
                    pre = path[i-1]
                    u_[(start,end)] = u_[(pre,start)] + 1

                x[k][(k,start,end)].setAttr('Start',1.0)
                u[(start,end)].setAttr('Start', u_[(start,end)])

                if isinstance(start,str):
                    tau[k][(k,pre,end)].setAttr('Start',1.0)

        m.update()
        print('initial solution updated')
        return

    def solve(self,
                init_solution = None,
                time_limit = 30
                ):
        
        print(' \n adding variables')
        self._add_variables()

        print(' \n adding constraints')
        self._add_constriants()

        print(' \n setting objective')
        self._set_objective()
        
        if init_solution:
            print(' \n using initial feasible solution')
            self._set_solution(init_solution)
        
        print(' \n solving ...')

        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()

        return
    
    
    def collect_numer_sol(self):
        '''
            this is run after model optimization
            to store the numerical value of the solution
            and remove gurobi components from the output
        '''
        x_numer = dict()
        for truck_ind, sol in self.variables['x'].items():
            x_numer[truck_ind] = {key:val.getAttr('x') for key,val in sol.items()}
        self.x_numer = x_numer
        
        u_numer = {key:val.getAttr('x') for key,val in self.variables['u'].items()}     
        self.u_numer = u_numer
        
        return

# a clean class for export

class GurobiSol(object):
    def __init__(self):
        self.x = None
        self.u = None
        
        self.obj = None
        
        self.G = None
        self.vrp = None
        
        return
    



# ===================================== utility functions ==========================================

def retrieve_path(sol:GurobiSol) -> dict[list]:
    ''' returns a dictionary of list
    key: truck id
    value: path as a list of nodes
    '''
    x = sol.x
    u = sol.u

    depot = sol.vrp.depot
    G = sol.G

    paths_dict = dict()
    num_trucks = len(x)
    
    for k in range(num_trucks):
        
        node = depot
        path = [depot]
        order_seq = []
        node_edge_dict = defaultdict(list)

        while True:
            
            # print(node)
            # record outgoing edges from the node
            if node not in node_edge_dict:
                for next_node in G.adj[node]:
                    x_i = x[k][(k, node, next_node)]
                    u_i = u[(node, next_node)]
                    if x_i > 0.5:
                        # print(k,node,next_node)
                        heapq.heappush(node_edge_dict[node], [u_i,next_node])
                        # print(node_edge_dict[node])
                        # if not isinstance(next_node,str): # do not add dummy
                        #     path.append(next_node)
                        # if next_node not in path:
                        #     node = next_node
                        #     break
                        
            if node_edge_dict[node]:
                u_next,next_node = heapq.heappop(node_edge_dict[node])
                order_seq.append(u_next)
                path.append(next_node)
                
                node = next_node
            else:
                if node == depot:
                    # this truck is not used at all
                    break
                raise ValueError(f'The current node {node} no longer leads to another node, check solution')
            
            if next_node == depot:
                # finished constructing path
                break
        
        path_no_dummy = [node for node in path if not isinstance(node,str)]
        paths_dict[k] = path_no_dummy
        
    return paths_dict


# %% external call function
def solve(vrp:TDVRP, G:nx.DiGraph,
            init_solution = None,
            time_limit = 30) -> dict:
    ''' a wrapper function
    '''
    g = GurobiSolve(vrp,G)
    g.solve(init_solution,time_limit)
    
    if gp.GRB.OPTIMAL == 3: #infeasible
        raise ValueError('The model is infeasible, skipping following actions')
    
    g.collect_numer_sol()   
    
    # collect output
    sol = GurobiSol()
    sol.x = g.x_numer
    sol.u = g.u_numer
    sol.obj = g.model.ObjVal
    sol.G = g.G
    sol.vrp = g.vrp
     
    return sol
