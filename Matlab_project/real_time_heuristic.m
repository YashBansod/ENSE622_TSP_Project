function [node_list, cost_mat, edge_table] = real_time_heuristic(cost_csv_path, node_csv_path)
%REAL_TIME_HEURISTIC Summary of this function goes here
%   Detailed explanation goes here

%% Parse the file containing graph information
% Parse the nodes in the graph used TSP
node_mat = readmatrix(node_csv_path);
node_list = round(node_mat(:, 1))';

% Parse the graph topology
edge_table = readmatrix(cost_csv_path);
edge_table(:, 1:2) = round(edge_table(:, 1:2));

%% Data restructuring

idxs = npermute2(1:length(node_list));
cost_mat = zeros(length(node_list));
for index = 1:size(edge_table, 1)
   cost_mat(idxs(index, 1), idxs(index, 2)) = edge_table(index, 3); 
end

end

