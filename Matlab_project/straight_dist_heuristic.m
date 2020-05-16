function [node_list, cost_mat, edge_table] = straight_dist_heuristic(node_csv_path)
%STRAIGHT_DIST_HEURISTIC Summary of this function goes here
%   Detailed explanation goes here

%% Parse the file containing graph information
% Parse the nodes in the graph used TSP
node_mat = readmatrix(node_csv_path);
node_list = round(node_mat(:, 1))';

x_cord = node_mat(:, 2)';
y_cord = node_mat(:, 3)';

%% Distance Computation
% Compute the distance matrix
x_ind_mat = repmat(x_cord, length(x_cord), 1);
x_ind_mat = x_ind_mat - x_cord';

y_ind_mat = repmat(y_cord, length(y_cord), 1);
y_ind_mat = y_ind_mat - y_cord';

cost_mat = hypot(x_ind_mat, y_ind_mat);

edge_table = npermute2(node_list);
edge_table(:, 3) = zeros(size(edge_table, 1), 1);

idxs = npermute2(1:length(node_list));
for index = 1:size(edge_table, 1)
   edge_table(index, 3) = cost_mat(idxs(index, 1), idxs(index, 2)); 
end

end

