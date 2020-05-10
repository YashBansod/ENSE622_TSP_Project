# !/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function

from map_data_handler import ParseCarlaMap

map_dir = "/home/yash/Downloads/CARLA_0.9.9/PythonAPI/examples/Recordings_3/"

print("Reading map from %s ..." % map_dir)

map_obj = ParseCarlaMap(map_dir)
# map_obj.plot_waypoints(block=True)
# map_obj.plot_spawnpoints(block=True)
# map_obj.plot_topology(block=True)
map_obj.write_toplogy_ids_as_csv(fname="town3_ids.csv")
map_obj.plot_all(block=True)
