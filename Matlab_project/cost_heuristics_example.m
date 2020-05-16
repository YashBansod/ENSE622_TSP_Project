%%
%   Author: Yash Bansod
%   Date: 16th May, 2020  
%   Cost_Heuristics_Example
%
% GitHub: <https://github.com/YashBansod>

%% Clear the environment and the command line
clear;
clc;
close all;

%% Example Usage

[node_list_1, cost_mat_1, edge_table_1] = straight_dist_heuristic("town1_nodes.csv");
[node_list_2, cost_mat_2, edge_table_2] = road_dist_heuristic("town1_ids.csv", "town1_nodes.csv");
[node_list_3, cost_mat_3, edge_table_3] = real_time_heuristic("Carla_01_normal.csv", "town1_nodes.csv");