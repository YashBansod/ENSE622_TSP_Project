# !/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function

from map_data_handler import ParseCarlaMap

recording_number = 1

# map_dir = "/home/yash/Downloads/CARLA_0.9.9/PythonAPI/examples/Recordings_%d/" % recording_number
map_dir = "D:\\CARLA\\WindowsNoEditor\\PythonAPI\\examples\\Recordings_%d\\" % recording_number
print("Reading map from %s ..." % map_dir)

map_obj = ParseCarlaMap(map_dir)

# map_obj.plot_waypoints(block=True)
# map_obj.plot_spawnpoints(block=True)
# map_obj.plot_topology(block=True)

map_obj.write_toplogy_ids_as_csv(fname="town%d_ids.csv" % recording_number)
# map_obj.plot_all(block=True)

node_indices = [46, 81, 132, 10, 40, 14, 110, 71, 58, 4]
map_obj.write_nodes_as_csv(node_indices=node_indices, fname="town%d_nodes.csv" % recording_number)
