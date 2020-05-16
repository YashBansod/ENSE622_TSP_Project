%%
%   Author: Yash Bansod
%   Date: 2nd May, 2020  
%   TSP Attemp 1
%
% GitHub: <https://github.com/YashBansod>

%% Clear the environment and the command line
clear;
close all;
clc;

%% Define the input parmeters
num_nodes = 10;                                 % Number of Nodes
node_labels = string(1:num_nodes);              % Node Labels
map_size = [50, 50];                            % [X, Y] Size

%% Define the environment data

% Create 'num_stops' random node coordinates
x_ind = randi(map_size(1), 1, num_nodes);
y_ind = randi(map_size(2), 1, num_nodes);

% Make sure the random node coordinates are unique
for curr_ind = 2:num_nodes
    for ind = 1:curr_ind - 1
        if (x_ind(curr_ind) == x_ind(ind) && y_ind(curr_ind) == y_ind(ind))
            x_ind(curr_ind) = randi(map_size(1));
            y_ind(curr_ind) = randi(map_size(2));
            ind = 1;
        end
    end 
end

%% Computations on environment data

% Compute the distance matrix
x_ind_mat = repmat(x_ind, num_nodes, 1);
x_ind_mat = x_ind_mat - x_ind';

y_ind_mat = repmat(y_ind, num_nodes, 1);
y_ind_mat = y_ind_mat - y_ind';

dist_mat = hypot(x_ind_mat, y_ind_mat);

% Create a dense graph of the nodes
idxs = nchoosek(1:num_nodes, 2);
map_graph = graph(idxs(:,1),idxs(:,2));

%% Plot the graph for visual confirmation
figure(1);
rectangle('Position', [0 0 map_size(1) map_size(2)]);
hold on;
grid on;
graph_plot = plot(map_graph, 'XData', x_ind', 'YData', y_ind',...
    'LineStyle', ':', 'NodeLabel', node_labels);
% hold off;
title('Environment');
xlabel('X-Coordinate');
ylabel('Y-Coordinate');

%% Define the Integer Linear Programming constraints

% Define the objective function
f = nonzeros(triu(dist_mat)');

% Constraint matrix and vector (A*X <= B)
A = [];
B = [];

Aeq = zeros(num_nodes, map_graph.numedges);
for ind = 1:num_nodes
    % List the trips starting or ending at node ind
    Aeq(ind, :) = sum((idxs == ind), 2)';
end
beq = 2 * ones(num_nodes, 1);

int_con = 1:map_graph.numedges;
lb = zeros(map_graph.numedges, 1);
ub = ones(map_graph.numedges, 1);

%% Optimize using Integer Linear Programming

% Define the optimizer options
opts = optimoptions('intlinprog', 'Display', 'off');
% Compute the solution using linear programming
[tsp, fval] = intlinprog(f, int_con, A, B, Aeq, beq, lb, ub, opts);

% Create a graph for the TSP
tsp = logical(round(tsp));
sol_graph = graph(idxs(tsp, 1), idxs(tsp, 2));


%% Plot the TSP for visual notification
highlight(graph_plot, sol_graph, 'LineStyle', '-', 'EdgeColor', 'r');
hold off;
title('Environment with TSP tour');
