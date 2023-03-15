import numpy as np
from utils.general import sample_box

random_seed = 0

# graph_bounding_box
# north, south, east, west
bounding_box = [10000, 0,10000,0]
depot_coord = (0,0)

# randomly genarate demand in rectangles
road_spacing = 200
demand_centers = [(2000,2000),
                  (8000,2000),
                  (5000,5000),
                  (2000,8000),
                  (8000,8000)]

# demand_centers = sample_box(10,bounding_box)

demand_std = [500] * len(demand_centers)
# number_of_demand = [20] * len(demand_centers)

number_of_demand = np.random.randint(100,200,len(demand_centers))



# truck parameters
truck_capacity = [500,400,300,200,200] * 50
truck_cost = [30000,20000,20000,15000,15000] * 50
num_trucks = len(truck_capacity)

truck_capacity = dict(zip(range(num_trucks),truck_capacity))
truck_cost = dict(zip(range(num_trucks),truck_cost))
