function [W, H] = normW(W_orig, H_orig)
    % nomalize W so that each col sums to 1
    s = sum(abs(W_orig));
    W = W_orig ./ s;
    % scale H accordingly
    H = H_orig .* s';
end
