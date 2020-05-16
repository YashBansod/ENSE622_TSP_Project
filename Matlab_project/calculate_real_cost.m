function [cost] = calculate_real_cost(tsp_tour, aggressive_cost_mat, normal_tl_cost_mat, cautious_tl_cost_mat)
%CALCULATE_REAL_COST Summary of this function goes here
%   Detailed explanation goes here
    tsp_tour_start = tsp_tour;
    tsp_tour_end = circshift(tsp_tour, -1);
    cost = 0;
    
for index = 1:length(tsp_tour_start)
    row_ind = tsp_tour_start(index);
    col_ind = tsp_tour_end(index);
    
    agg_cost = aggressive_cost_mat(row_ind, col_ind);
    nor_tl_cost = normal_tl_cost_mat(row_ind, col_ind);
    cau_tl_cost = cautious_tl_cost_mat(row_ind, col_ind);
    
    cost_list = sort([agg_cost, nor_tl_cost, cau_tl_cost]);
    pd = makedist('Triangular','a',cost_list(1), 'b', cost_list(2), ...
        'c', cost_list(3));
    cost = cost + random(pd);
end

