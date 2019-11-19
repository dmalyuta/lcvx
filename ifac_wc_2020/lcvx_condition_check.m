% Verify that lossless convexification conditions are satisfied for IFAC 2020
% numerical example.
%
% D. Malyuta -- ACL, University of Washington
% B. Acikmese -- ACL, University of Washington
% 
% Copyright 2019 University of Washington. All rights reserved.

addpath(genpath('../lib'));

%% Parameters

n = 4;
m = 2;
omega = 2*pi/(24*3600+39*60+35); % [rad/s] Mars spin
g = 3.71; % [m/s^2] Mars surface gravity
S = [0,1;-1,0];
A = [zeros(2),eye(2);omega^2*eye(2),2*omega*S];
B = [zeros(2);eye(2)];
D = [eye(2);zeros(2)];
theta = [120,10]*pi/180;
rho2 = [8,12];
wx_factor = 1e-3;
tf_max = 100;
gamma_gs = 10*pi/180;
h0 = [650,750,850,1000,1200];
zero_tol = 1e-7;

%% Pre-process parameters

A_ = -A';
B_ = D;
C_ = B';
D_ = zeros(2);

ey = [0;1];

% Compute the state penalty gradient
xi_max = tf_max*rho2(2);
wx = wx_factor*xi_max;
grad_r_l = cell(0);
for i = 1:numel(h0)
    grad_r_l{end+1} = [tan(gamma_gs);1]*wx/h0(i); %#ok
    grad_r_l{end+1} = [-tan(gamma_gs);1]*wx/h0(i); %#ok
end

%% Condition 1

WUS = weak_unobsv_sub(A_,B_,C_,D_);
if ~isempty(WUS)
    error('Condition 1 fails');
end

%% Condition 2 and 3 with zeta=0

Rot = @(ang) [cos(ang),-sin(ang);sin(ang),cos(ang)];

% equivalence manifold
alpha_equiv = acos(rho2(1)/rho2(2));
assert(alpha_equiv<=(theta(1)/2-theta(2)/2),...
       'equivalence line outside U_1\U_2');

cone_side_dirs = {Rot(theta(1)/2)*ey,Rot(-theta(1)/2)*ey,...
                  Rot(theta(2)/2)*ey,Rot(-theta(2)/2)*ey,...
                  Rot(alpha_equiv+theta(2)/2)*ey,...
                  Rot(pi/2)*Rot(-(alpha_equiv+theta(2)/2))*ey};

A_aug = [A_,B_;zeros(m,n+m)];
B_aug = zeros(n+m,1);
C_aug = [C_,zeros(m,m)];
D_aug = zeros(m,1);

for i = 1:numel(cone_side_dirs)
    v = cone_side_dirs{i};
    WUS = weak_unobsv_sub(A_aug,B_aug,v'*C_aug*A_aug,v'*D_aug);
    if ~isempty(WUS)
        for k = 1:size(WUS,2)
            col_ = WUS(:,k);
            z_dot = A_aug*col_;
            if norm(z_dot)>zero_tol
                disp(norm(z_dot))
                % This is a moving y, i.e. non-zero velocity
                if i<=4
                    error('Condition 2 with zeta=0 fails');
                else
                    error('Condition 3 with zeta=0 fails');
                end
            end
        end
        % range_input = WUS(n+1:end,:);
        % for j = 1:numel(grad_r_l)
        %     input = grad_r_l{j};
        %     lin_comb = range_input\input;
        %     if any(abs(range_input*lin_comb-input)>eps)
        %         % WUS does not contain our constant input, grad_x(ell)
        %         continue
        %     end
        %     % WUS contains our constant input grad_x(ell)
        %     % Check that this is a static case (i.e. y not changing)
        %     N = null(range_input);
        %     cols_to_check = [lin_comb,N];
        %     for k = 1:size(cols_to_check,2)
        %         col_ = cols_to_check(:,k);
        %         z_dot = A_aug*WUS*col_;
        %         if abs(z_dot)>eps
        %             % This is a moving y, i.e. non-zero velocity
        %             if i<=4
        %                 error('Condition 2 fails');
        %             else
        %                 error('Condition 3 with zeta=0 fails');
        %             end
        %         end
        %     end
        % end
    end
end

%% Condition 2 with zeta=1, y \in U2
% Check the "orbiting case" norm(y)=1 for non-trivial duration

[V_,Deig_] = eig(A_aug);

% Orbiting mode shapes
eigvals = diag(Deig_);
orbiting_mode_shapes = V_(:,abs(real(eigvals))<sqrt(eps) & ...
                          abs(imag(eigvals))>sqrt(eps));
range_input = orbiting_mode_shapes(n+1:end,:);
if rank(range_input)~=0
    for j = 1:numel(grad_r_l)
        input = grad_r_l{j};
        lin_comb = range_input\input;
        if any(abs(range_input*lin_comb-input)>zero_tol)
            % range_input does not contain our constant input, grad_x(ell)
            continue
        end
        disp(abs(range_input*lin_comb-input))
        error('Condition 3 with zeta=1 orbiting case (y \in U2) fails');
    end
end

% Case that remains: non-orbiting equivalence case
% (proj_U1(y)-1)*rho2(1)=(proj_U2(y)-1)*rho2(2) for finite duration
%
% So far, not able to check this except a-posteriori solving the problem...
