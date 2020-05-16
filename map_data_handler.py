#!/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function

import os
import glob
import pickle
import numpy as np
from matplotlib import pyplot as plt


# ******************************************    Function Definitions        ****************************************** #
def degree_180_to_m180(degrees):
    degrees = ((degrees + 180.0) % 360.0) - 180.0
    return degrees


# ******************************************    Function Declaration End    ****************************************** #


# ******************************************    Class Declaration Start     ****************************************** #
class MapWaypointWriter(object):

    def __init__(self, map_obj, wayp_dist=None, spawn_pts=False, topology=False):

        self._root_path = os.getcwd() + "/Recordings/"
        if not os.path.exists(self._root_path):
            os.makedirs(self._root_path)

        if wayp_dist:
            self._waypoints = map_obj.generate_waypoints(wayp_dist)

            x_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)
            y_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)
            z_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)

            yaw_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)
            pitch_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)
            roll_arr = np.zeros(shape=len(self._waypoints), dtype=np.float32)

            road_arr = np.zeros(shape=len(self._waypoints), dtype=np.int16)
            lane_arr = np.zeros(shape=len(self._waypoints), dtype=np.int16)

            for index in range(len(self._waypoints)):
                waypoint = self._waypoints[index]

                x_arr[index] = waypoint.transform.location.x
                y_arr[index] = waypoint.transform.location.y
                z_arr[index] = waypoint.transform.location.z

                yaw_arr[index] = waypoint.transform.rotation.yaw
                pitch_arr[index] = waypoint.transform.rotation.pitch
                roll_arr[index] = waypoint.transform.rotation.roll

                road_arr[index] = waypoint.road_id
                lane_arr[index] = waypoint.lane_id

            yaw_arr = degree_180_to_m180(yaw_arr)
            pitch_arr = degree_180_to_m180(pitch_arr)
            roll_arr = degree_180_to_m180(roll_arr)

            wayp_dict = {"location": {"x": x_arr, "y": y_arr, "z": z_arr},
                         "rotation": {"yaw": yaw_arr, "pitch": pitch_arr, "roll": roll_arr},
                         "road_id": road_arr, "lane_id": lane_arr}

            pickle.dump(wayp_dict, open(self._root_path + map_obj.name + "_waypoints.pkl", "wb"))

        if spawn_pts:
            self._spawn_points = map_obj.get_spawn_points()
            x_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)
            y_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)
            z_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)

            yaw_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)
            pitch_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)
            roll_arr = np.zeros(shape=len(self._spawn_points), dtype=np.float32)

            for index in range(len(self._spawn_points)):
                spawn_point = self._spawn_points[index]

                x_arr[index] = spawn_point.location.x
                y_arr[index] = spawn_point.location.y
                z_arr[index] = spawn_point.location.z

                yaw_arr[index] = spawn_point.rotation.yaw
                pitch_arr[index] = spawn_point.rotation.pitch
                roll_arr[index] = spawn_point.rotation.roll

            yaw_arr = degree_180_to_m180(yaw_arr)
            pitch_arr = degree_180_to_m180(pitch_arr)
            roll_arr = degree_180_to_m180(roll_arr)

            spawn_dict = {"location": {"x": x_arr, "y": y_arr, "z": z_arr},
                          "rotation": {"yaw": yaw_arr, "pitch": pitch_arr, "roll": roll_arr}}

            pickle.dump(spawn_dict, open(self._root_path + map_obj.name + "_spawnpoints.pkl", "wb"))

        if topology:
            self._topology = map_obj.get_topology()

            id_1 = np.zeros(shape=len(self._topology), dtype=np.int16)

            x_arr = np.zeros(shape=len(self._topology), dtype=np.float32)
            y_arr = np.zeros(shape=len(self._topology), dtype=np.float32)
            z_arr = np.zeros(shape=len(self._topology), dtype=np.float32)

            yaw_arr = np.zeros(shape=len(self._topology), dtype=np.float32)
            pitch_arr = np.zeros(shape=len(self._topology), dtype=np.float32)
            roll_arr = np.zeros(shape=len(self._topology), dtype=np.float32)

            road_arr = np.zeros(shape=len(self._waypoints), dtype=np.int16)
            lane_arr = np.zeros(shape=len(self._waypoints), dtype=np.int16)

            id_2 = np.zeros(shape=len(self._topology), dtype=np.int16)

            x_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)
            y_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)
            z_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)

            yaw_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)
            pitch_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)
            roll_arr_2 = np.zeros(shape=len(self._topology), dtype=np.float32)

            road_arr_2 = np.zeros(shape=len(self._topology), dtype=np.int16)
            lane_arr_2 = np.zeros(shape=len(self._topology), dtype=np.int16)

            id_dict = {}
            node_dict = {"location": {"x": [], "y": [], "z": []},
                         "rotation": {"yaw": [], "pitch": [], "roll": []}}
            id_count = 0

            for index in range(len(self._topology)):
                wp_from = self._topology[index][0]
                wp_to = self._topology[index][1]

                if wp_from.id in id_dict:
                    id_from = id_dict[wp_from.id]
                else:
                    id_dict[wp_from.id] = id_count
                    id_from = id_count
                    node_dict['location']['x'].append(wp_from.transform.location.x)
                    node_dict['location']['y'].append(wp_from.transform.location.y)
                    node_dict['location']['z'].append(wp_from.transform.location.z)
                    node_dict['rotation']['yaw'].append(degree_180_to_m180(wp_from.transform.rotation.yaw))
                    node_dict['rotation']['pitch'].append(degree_180_to_m180(wp_from.transform.rotation.pitch))
                    node_dict['rotation']['roll'].append(degree_180_to_m180(wp_from.transform.rotation.roll))
                    id_count += 1

                id_1[index] = id_from

                x_arr[index] = wp_from.transform.location.x
                y_arr[index] = wp_from.transform.location.y
                z_arr[index] = wp_from.transform.location.z

                yaw_arr[index] = wp_from.transform.rotation.yaw
                pitch_arr[index] = wp_from.transform.rotation.pitch
                roll_arr[index] = wp_from.transform.rotation.roll

                road_arr[index] = wp_from.road_id
                lane_arr[index] = wp_from.lane_id

                if wp_to.id in id_dict:
                    id_to = id_dict[wp_to.id]
                else:
                    id_dict[wp_to.id] = id_count
                    id_to = id_count
                    node_dict['location']['x'].append(wp_to.transform.location.x)
                    node_dict['location']['y'].append(wp_to.transform.location.y)
                    node_dict['location']['z'].append(wp_to.transform.location.z)
                    node_dict['rotation']['yaw'].append(degree_180_to_m180(wp_to.transform.rotation.yaw))
                    node_dict['rotation']['pitch'].append(degree_180_to_m180(wp_to.transform.rotation.pitch))
                    node_dict['rotation']['roll'].append(degree_180_to_m180(wp_to.transform.rotation.roll))
                    id_count += 1

                id_2[index] = id_to

                x_arr_2[index] = wp_to.transform.location.x
                y_arr_2[index] = wp_to.transform.location.y
                z_arr_2[index] = wp_to.transform.location.z

                yaw_arr_2[index] = wp_to.transform.rotation.yaw
                pitch_arr_2[index] = wp_to.transform.rotation.pitch
                roll_arr_2[index] = wp_to.transform.rotation.roll

                road_arr_2[index] = wp_to.road_id
                lane_arr_2[index] = wp_to.lane_id

            yaw_arr = degree_180_to_m180(yaw_arr)
            pitch_arr = degree_180_to_m180(pitch_arr)
            roll_arr = degree_180_to_m180(roll_arr)

            yaw_arr_2 = degree_180_to_m180(yaw_arr_2)
            pitch_arr_2 = degree_180_to_m180(pitch_arr_2)
            roll_arr_2 = degree_180_to_m180(roll_arr_2)

            topology_dict = {"id_from": id_1, "location_from": {"x": x_arr, "y": y_arr, "z": z_arr},
                             "rotation_from": {"yaw": yaw_arr, "pitch": pitch_arr, "roll": roll_arr},
                             "road_id_from": road_arr, "lane_id_from": lane_arr,
                             "id_to": id_2, "location_to": {"x": x_arr_2, "y": y_arr_2, "z": z_arr_2},
                             "rotation_to": {"yaw": yaw_arr_2, "pitch": pitch_arr_2, "roll": roll_arr_2},
                             "road_id_to": road_arr_2, "lane_id_to": lane_arr_2}

            pickle.dump(topology_dict, open(self._root_path + map_obj.name + "_topology.pkl", "wb"))
            pickle.dump(node_dict, open(self._root_path + map_obj.name + "_node_dict.pkl", "wb"))


# ******************************************    Class Declaration End       ****************************************** #


# ******************************************    Class Declaration Start     ****************************************** #
class ParseCarlaMap(object):

    def __init__(self, root_path):
        self._root_path = root_path
        self._wp_pkl_path = glob.glob(self._root_path + "Town*_waypoints.pkl")[0]
        self._sp_pkl_path = glob.glob(self._root_path + "Town*_spawnpoints.pkl")[0]
        self._tp_pkl_path = glob.glob(self._root_path + "Town*_topology.pkl")[0]
        self._nd_pkl_path = glob.glob(self._root_path + "Town*_node_dict.pkl")[0]

        self.waypoint_data = pickle.load(open(self._wp_pkl_path, "rb"))
        self.spawnpoint_data = pickle.load(open(self._sp_pkl_path, "rb"))
        self.topology_data = pickle.load(open(self._tp_pkl_path, "rb"))
        self.node_dict = pickle.load(open(self._nd_pkl_path, "rb"))

    # ******************************        Class Method Declaration        ****************************************** #
    def write_nodes_as_csv(self, node_indices, fname):
        node_table = np.zeros(shape=(len(node_indices), 3))
        node_table[:, 0] = node_indices
        node_table[:, 1] = np.array(self.node_dict["location"]["x"])[node_indices]
        node_table[:, 2] = np.array(self.node_dict["location"]["y"])[node_indices]

        file_path = self._root_path + fname
        # noinspection PyTypeChecker
        np.savetxt(file_path, node_table, fmt="%3d,%3.3f,%3.3f")

    # ******************************        Class Method Declaration        ****************************************** #
    def eu_dist(self, x_cord, y_cord):
        delta_x = self.waypoint_data["location"]["x"] - x_cord
        delta_y = self.waypoint_data["location"]["y"] - y_cord

        distance = np.sqrt(np.square(delta_x) + np.square(delta_y))
        return distance

    # ******************************        Class Method Declaration        ****************************************** #
    def nearest_wp_index(self, x_cord, y_cord):
        distance = self.eu_dist(x_cord, y_cord)
        return np.argmin(distance)

    # ******************************        Class Method Declaration        ****************************************** #
    def write_toplogy_ids_as_csv(self, fname):
        topology = np.zeros(shape=(len(self.topology_data["location_from"]["x"]), 3))
        topology[:, 0] = self.topology_data["id_from"]
        topology[:, 1] = self.topology_data["id_to"]
        topology[:, 2] = np.sqrt(
            np.square(self.topology_data["location_from"]["x"] - self.topology_data["location_to"]["x"]) +
            np.square(self.topology_data["location_from"]["y"] - self.topology_data["location_to"]["y"])
        )
        file_path = self._root_path + fname
        # noinspection PyTypeChecker
        np.savetxt(file_path, topology, fmt="%3d,%3d,%3.3f")

    # ******************************        Class Method Declaration        ****************************************** #
    def nearest_wp_cord(self, x_cord, y_cord):
        distance = self.eu_dist(x_cord, y_cord)
        index = np.argmin(distance)
        return self.waypoint_data["location"]["x"][index], self.waypoint_data["location"]["y"][index]

    # ******************************        Class Method Declaration        ****************************************** #
    def get_waypoint(self, index):

        waypoint = {"x": self.waypoint_data["location"]["x"][index],
                    "y": self.waypoint_data["location"]["y"][index],
                    "z": self.waypoint_data["location"]["z"][index],

                    "yaw": self.waypoint_data["rotation"]["yaw"][index],
                    "pitch": self.waypoint_data["rotation"]["pitch"][index],
                    "roll": self.waypoint_data["rotation"]["roll"][index],

                    "road_id": self.waypoint_data["road_id"][index],
                    "lane_id": self.waypoint_data["lane_id"][index]}
        return waypoint

    # ******************************        Class Method Declaration        ****************************************** #
    def nearest_wp(self, x_cord, y_cord):
        distance = self.eu_dist(x_cord, y_cord)
        index = np.argmin(distance)
        waypoint = self.get_waypoint(index)

        return waypoint

    # ******************************        Class Method Declaration        ****************************************** #
    def plot_waypoints(self, block=False):

        plt.figure(figsize=(20, 20))
        plt.scatter(self.waypoint_data["location"]["x"], self.waypoint_data["location"]["y"], np.pi * 5)
        plt.title("CARLA Map %s Waypoints" %
                  self._wp_pkl_path[self._wp_pkl_path.find('/Town') + 1: self._wp_pkl_path.find('_waypoints.pkl')])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.grid()
        plt.show(block=block)

    # ******************************        Class Method Declaration        ****************************************** #
    def plot_spawnpoints(self, block=False):
        plt.figure(figsize=(20, 20))
        plt.scatter(self.spawnpoint_data["location"]["x"], self.spawnpoint_data["location"]["y"], np.pi * 10)
        plt.title("CARLA Map %s Spawnpoints" %
                  self._sp_pkl_path[self._sp_pkl_path.find('/Town') + 1: self._sp_pkl_path.find('_spawnpoints.pkl')])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.grid()
        plt.show(block=block)

    # ******************************        Class Method Declaration        ****************************************** #
    def plot_topology(self, block=False):
        plt.figure(figsize=(20, 20))
        plt.scatter(self.topology_data["location_from"]["x"], self.topology_data["location_from"]["y"],
                    np.pi * 20, c='r', marker='o')
        plt.scatter(self.topology_data["location_to"]["x"], self.topology_data["location_to"]["y"],
                    np.pi * 10, c='b', marker='x')
        for index in range(len(self.topology_data["location_from"]["x"])):
            plt.arrow(self.topology_data["location_from"]["x"][index], self.topology_data["location_from"]["y"][index],
                      self.topology_data["location_to"]["x"][index] - self.topology_data["location_from"]["x"][index],
                      self.topology_data["location_to"]["y"][index] - self.topology_data["location_from"]["y"][index],
                      width=0.25, head_width=1)
        plt.title("CARLA Map %s Topology" %
                  self._tp_pkl_path[self._tp_pkl_path.find('/Town') + 1: self._tp_pkl_path.find('_topology.pkl')])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.grid()
        plt.show(block=block)

    # ******************************        Class Method Declaration        ****************************************** #
    def plot_all(self, block=False):
        plt.figure(figsize=(20, 20))
        # plt.scatter(self.waypoint_data["location"]["x"], self.waypoint_data["location"]["y"], np.pi * 3,
        #             c='g', marker='+')
        plt.scatter(self.spawnpoint_data["location"]["x"], self.spawnpoint_data["location"]["y"], np.pi * 50,
                    c='m', marker='1')
        plt.scatter(self.topology_data["location_from"]["x"], self.topology_data["location_from"]["y"],
                    np.pi * 20, c='r', marker='x')
        plt.scatter(self.topology_data["location_to"]["x"], self.topology_data["location_to"]["y"],
                    np.pi * 10, c='b', marker='*')
        for index in range(len(self.topology_data["location_from"]["x"])):
            plt.arrow(self.topology_data["location_from"]["x"][index],
                      self.topology_data["location_from"]["y"][index],
                      self.topology_data["location_to"]["x"][index] - self.topology_data["location_from"]["x"][
                          index],
                      self.topology_data["location_to"]["y"][index] - self.topology_data["location_from"]["y"][
                          index],
                      width=0.25, head_width=2)
            plt.text(self.topology_data["location_from"]["x"][index], self.topology_data["location_from"]["y"][index],
                     self.topology_data["id_from"][index], color='r', fontsize=12)
            # plt.text(self.topology_data["location_to"]["x"][index], self.topology_data["location_to"]["y"][index],
            #          self.topology_data["id_to"][index], color='b', fontsize=12)
        plt.title("CARLA Map %s" %
                  self._tp_pkl_path[self._tp_pkl_path.find('/Town') + 1: self._tp_pkl_path.find('_topology.pkl')])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.grid()
        plt.show(block=block)

# ******************************************    Class Declaration End       ****************************************** #
