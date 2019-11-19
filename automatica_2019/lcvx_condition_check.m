% Verify that lossless convexification conditions are satisfied for Automatica
% numerical example.
%
% D. Malyuta -- ACL, University of Washington
% B. Acikmese -- ACL, University of Washington
% 
% Copyright 2019 University of Washington. All rights reserved.

addpath(genpath('../lib'));

%% Parameters

n = 6;
m = 3;
omega = [0,0,2]'*pi/60; % [rad/s] Space station spin
S = @(w) [0,-w(3),w(2);w(3),0,-w(1);-w(2),w(1),0];
A = [zeros(3),eye(3);-S(omega)^2,-2*S(omega)];
B = [zeros(3);eye(3)];
nhat = {[ 0.49240388, -0.64278761,  0.58682409]',...
        [0.49240388, 0.64278761, 0.58682409]',...
        [-0.49240388,  0.64278761,  0.58682409]',...
        [-0.49240388, -0.64278761,  0.58682409]',...
        [ 0.4330127, -0.5      , -0.75     ]',...
        [ 0.4330127,  0.5      , -0.75     ]',...
        [-0.4330127,  0.5      , -0.75     ]',...
        [-0.4330127, -0.5      , -0.75     ]',...
        [-0., -1., -0.]',...
        [-0.,  1., -0.]',...
        [ 1., -0., -0.]',...
        [-1., -0., -0.]'};
K = 4;
M = 12;
zero_tol = 1e-14;

%% Pre-process parameters

A_ = -A';
B_ = zeros(n,1);
C_ = B';
D_ = zeros(m,1);

%% Condition 1

WUS = weak_unobsv_sub(A_,B_,C_,D_);
if ~isempty(WUS)
    error('Condition 1 fails');
end

%% Condition 2

for i = 1:numel(nhat)
    % Check if case (a) holds
    WUS = weak_unobsv_sub(A_,B_,nhat{i}'*C_,nhat{i}'*D_);
    if ~isempty(WUS)
        % Check that case (b) holds
        y_ = range((C_*WUS)')';
        assert(size(y_,2)==1)
        for k = 1:n-1
            tmp_ = range((C_*A_^k*WUS)')';
            assert(size(tmp_,2)==1)
            if norm(tmp_)>zero_tol && norm(tmp_-y_)>zero_tol
                error('Condition 2 fails');
            end
        end
        for sign_ = [1,-1]
            v_ = sign_*y_;
            proj = nan(M,1);
            for j = 1:M
                proj(j) = nhat{j}'*v_;
            end
            if sum(proj>0)<K
                error('Condition 2 fails');
            end
        end
    end
end

%% Condition 3

for i = 1:numel(nhat)
    for j = i+1:numel(nhat)
        % Check if case (a) holds
        n_ = nhat{i}-nhat{j};
        WUS = weak_unobsv_sub(A_,B_,n_'*C_,n_'*D_);
        if ~isempty(WUS)
            % Check that case (b) holds
            y_ = range((C_*WUS)')';
            assert(size(y_,2)==1)
            for k = 1:n-1
                tmp_ = range((C_*A_^k*WUS)')';
                assert(size(tmp_,2)==1)
                if norm(tmp_)>zero_tol && norm(tmp_-y_)>zero_tol
                    error('Condition 3 fails');
                end
            end
            for sign_ = [1,-1]
                v_ = sign_*y_;
                proj = nan(M,1);
                for k = 1:M
                    proj(k) = nhat{k}'*v_;
                end
                if sum(proj>0)<K
                    error('Condition 3 fails');
                end
            end
        end
    end
end
