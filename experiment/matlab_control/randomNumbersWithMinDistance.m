function selected_numbers = randomNumbersWithMinDistance(n, k, m)
    if k * (m - 1) >= n
        error('Cannot select k numbers with a minimal distance m in the given range.');
    end

    selected_numbers = zeros(1, k);

    for i = 1:k
        % Generate a random number between 1 and n
        random_num = randi(n);

        % Check if the generated number is too close to any previously selected number
        while any(abs(selected_numbers - random_num) < m)
            random_num = randi(n);
        end

        selected_numbers(i) = random_num;
    end
end
