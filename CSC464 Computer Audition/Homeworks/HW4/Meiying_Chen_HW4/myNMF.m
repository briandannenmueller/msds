function [W, H, KL] = myNMF(V, r, nIter, initW, initH, bUpdateW, bUpdateH)
% Implementation of the multiplicative update rule for NMF with K-L
% divergence. W should always be normalized so that each column of W sums
% to 1. Scale H accordingly. This normalization step should be performed in
% each iteration, and in initialization as well. This is to deal with the
% non-uniqueness of NMF solutions.
%
% Input
%   - V         : a non-negative matrix (m*n) to be decomposed
%   - r         : #columns of W (i.e., #rows of H)
%   - nIter     : #iterations
%   - initW     : initial value of W (m*r) (default: a random non-negative matrix)
%   - initH     : initial value of H (r*n) (default: a random non-negative matrix)
%   - bUpdateW  : update W (bUpdateW==1) or not (bUpdateW==0) (default: 1)
%   - bUpdateH  : update H (bUpdateH==1) or not (bUpdateH==0) (default: 1)
%
% Output
%   - W         : learned W
%   - H         : learned H
%   - KL        : KL divergence after each iteration (a row vector with nIter+1
%               elements). KL(1) is the KL divergence using the initial W
%               and H.
%
% Author: Zhiyao Duan
% Created: 10/9/2013
% Last modified: 10/13/2019

[m,n] = size(V);
if nargin<7 bUpdateH=1; end
if nargin<6 bUpdateW=1; end
if nargin<5 initH = rand(r, n); end     % randomly initialize H
if nargin<4 initW = rand(m, r); end     % randomly initialize W
if r~=size(initW,2) || r~=size(initH,1)
    error('Parameter r and the size of W or H do not match!');
end

% Your implementation starts here...
    V = abs(V);
    [m, n] = size(V);
    [W, H] = normW(initW, initH);           % nomalize W
    KL = zeros(1, nIter);                   % init kl vector

    % multiplicative update for KL-divergence
    for k = 1:nIter
        WH = W*H;

        kl = sum(V .* log(V./WH) - V + WH, 'all');
        KL(k) = kl;

        %disp(k);

        if bUpdateW
            for i = 1:m
                for a = 1:r
                    HV_WH = 0;
                    for u = 1:n
                        HV_WH = HV_WH + H(a,u)*V(i,u)'/WH(i,u);
                    end 
                    W(i,a) = W(i,a) * HV_WH/sum(H(a,:));
                end
            end
        end

        if bUpdateH
            for a = 1:r
                for u = 1:n
                    WV_WH = 0;
                    for i = 1:m
                        WV_WH = WV_WH + W(i,a)'*V(i,u)/WH(i,u);
                    end
                    H(a,u) = H(a,u) *WV_WH/sum(W(:,a));
                end
            end
        end

        % norm W and scale H at each iteration
        [W, H] = normW(W, H);
    end

end















