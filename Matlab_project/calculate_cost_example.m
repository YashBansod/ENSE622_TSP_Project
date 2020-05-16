%%
%   Author: Yash Bansod
%   Date: 16th May, 2020  
%   Calculate_Cost_Example
%
% GitHub: <https://github.com/YashBansod>

%% Clear the environment and the command line
clear;
clc;
close all;

%% Example Usage
[~, agg_cost_mat, ~] = real_time_heuristic("Carla_01_aggressive.csv", "town1_nodes.csv");
[~, nor_tl_cost_mat, ~] = real_time_heuristic("Carla_01_normal_tl.csv", "town1_nodes.csv");
[~, cau_tl_cost_mat, ~] = real_time_heuristic("Carla_01_cautious_tl.csv", "town1_nodes.csv");

tsp_tour = [1	2	3	4	5	6	7	8	9	10];
num_samples = 5000;                 % Number of times to sample cost

cost_estimates = zeros(1, num_samples);

for index = 1:num_samples
    cost_estimates(index) = calculate_real_cost(tsp_tour, agg_cost_mat, nor_tl_cost_mat, cau_tl_cost_mat);
end

%% Plot results
figure(1)
num_hist_bins = 15;
histogram(cost_estimates, num_hist_bins);

%% Print the computation results

fprintf('Cost - Mean: %.2f\n', mean(cost_estimates));
fprintf('Cost - Standard Deviation: %.2f\n', std(cost_estimates))