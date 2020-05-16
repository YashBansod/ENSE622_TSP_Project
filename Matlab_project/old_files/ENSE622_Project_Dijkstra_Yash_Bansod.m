%%
%   Author: Yash Bansod
%   Date: 19th February, 2020  
%   Problem 1, Assignment 3 (ENSE622)
%
% GitHub: <https://github.com/YashBansod>

%% Clear the environment and the command line
clear;
clc;
close all;

%% Parse the file containing graph information
graph_mat = readmatrix("town3_ids.csv");
graph_mat(:, 1) = round(graph_mat(:, 1)) + 1;
graph_mat(:, 2) = round(graph_mat(:, 2)) + 1;
graph_mat(:, 3) = graph_mat(:, 3);

node_list = [25, 36, 670, 400, 570, 240, 200];

%% Define the graph

% Specify the edges and thier costs
e_start = graph_mat(:, 1);
e_stop  = graph_mat(:, 2);
e_cost  = graph_mat(:, 3);

% Create the graph
graph = digraph(e_start, e_stop, e_cost);


%% Calculate and plot the shortest path
edge_table = npermute2(node_list);
edge_table(:, 3) = zeros(size(edge_table, 1), 1);

for index = 1:size(edge_table, 1)
    [shortest_path, edge_table(index, 3)] = ...
        shortestpath(graph, edge_table(index, 1), edge_table(index, 2));
end
    
%% Print the computation results

disp("Completed edge table creation")