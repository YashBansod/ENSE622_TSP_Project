function tuples = npermute2(n)
%NPERMUTE2
    num_rows = (size(n, 2) * size(n, 2)) - size(n, 2);
    tuples = zeros(num_rows, 2);
    index = 1;
    for ind_1=1:size(n, 2)
        for ind_2=1:size(n, 2)
            if ind_2 == ind_1
                continue
            end
            tuples(index, 1) = n(ind_1);
            tuples(index, 2) = n(ind_2);
            index = index + 1;
        end
    end
end

