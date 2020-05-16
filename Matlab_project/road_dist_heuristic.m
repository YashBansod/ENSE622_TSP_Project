function [node_list, cost_mat, edge_table] = road_dist_heuristic(ids_csv_path, node_csv_path)
%ROAD_DIST_HEURISTIC Summary of this function goes here
%   Detailed explanation goes here

%% Parse the file containing graph information
% Parse the graph topology
graph_mat = readmatrix(ids_csv_path);
graph_mat(:, 1:2) = round(graph_mat(:, 1:2)) + 1;

% Parse the nodes in the graph used TSP
node_mat = readmatrix(node_csv_path);
node_list = round(node_mat(:, 1))' + 1;

%% Define the graph
% Specify the edges and thier costs
e_start = graph_mat(:, 1);
e_stop  = graph_mat(:, 2);
e_cost  = graph_mat(:, 3);

% Create the graph
graph = digraph(e_start, e_stop, e_cost);

%% Calculate the shortest path
edge_table = npermute2(node_list);
edge_table(:, 3) = zeros(size(edge_table, 1), 1);

idxs = npermute2(1:length(node_list));
cost_mat = zeros(length(node_list));

for ind = 1:size(edge_table, 1)
    [~, edge_table(ind, 3)] = ...
        shortestpath(graph, edge_table(ind, 1), edge_table(ind, 2));
    cost_mat(idxs(ind, 1), idxs(ind, 2)) = edge_table(ind, 3);
end

node_list = node_list - 1;
edge_table(:, 1:2) = edge_table(:, 1:2) - 1;

end

