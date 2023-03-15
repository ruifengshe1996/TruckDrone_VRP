random_seed = 0

# graph_bounding_box
bounding_box = [40.3273, 40.2820,-88.2135,-88.1166]
depot_coord = (-88.1896,40.31373)

# randomly genarate demand in rectangles
demand_centers = [(-88.1606,40.3127),
                  (-88.1457,40.3193),
                  (-88.1499,40.3074),
                  (-88.1321,40.3086),
                  (-88.158,40.2976),
                  (-88.1626,40.2861)]

# demand_centers = [(-88.1606,40.3127),
#                   (-88.1457,40.3193)]

num_centers = len(demand_centers)
number_of_demand = [150] * num_centers
demand_std = [210] * num_centers
# truck parameters
truck_capacity = [200,200,200,200,200,200]
truck_cost = [30000,20000,20000,15000,15000,15000]
num_trucks = len(truck_capacity)

truck_capacity = dict(zip(range(num_trucks),truck_capacity))
truck_cost = dict(zip(range(num_trucks),truck_cost))