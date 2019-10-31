% Verify that the system formed by the rocket landing dynamics together with the
% state integral penalty matrix Q, i.e. {A,B,Q,0}, is strongly
% controllable.
%
% D. Malyuta -- ACL, University of Washington
% B. Acikmese -- ACL, University of Washington
% 
% Copyright 2019 University of Washington. All rights reserved.

n = 4;
m = 2;
omega = 1;%2*pi/(24*3600+39*60+35); % [rad/s] Mars spin
g = 3.71; % [m/s^2] Mars surface gravity
S = [0,1;-1,0];
A = [zeros(2),eye(2);omega^2*eye(2),2*omega*S];
B = [zeros(2);eye(2)];
D = [eye(2);zeros(2)];

% Check Condition 1
V = weak_unobsv_sub(A',D,B',zeros(2));
if ~isempty(V)
    error('system has transmission zeroes');
else
    fprintf('{A'',D,B'',0} is strongly controllable\n');
end

% Check Conditions 2 and 3
% Parameters from landing_2mode.py
theta = [90,20];
rho2 = [8,12];
wx_factor = 1e-3;
tf_max = 100;
gamma_gs = 10*pi/180;
h0 = [650,750,850,1000,1200];
align_tol = 1e-3*pi/180; % tolerance for unobservability
% Compute the state penalty gradient
xi_max = tf_max*rho2(2);
wx = wx_factor*xi_max;
grad_r_l = nan(2,numel(h0));
for i = 1:size(grad_r_l,2)
    grad_r_l(:,i) = [tan(gamma_gs);1]*wx/h0(i);
end
% Compute line normals
% 1) normal cone facets
R = @(angle) [cosd(angle),-sind(angle);sind(angle),cosd(angle)];
ey = [0;1];
nhat = [R(theta(1)/2)*ey,R(-theta(1)/2)*ey,...
        R(theta(2)/2)*ey,R(-theta(2)/2)*ey];
% 2) equivalence manifold facets
r_ = rho2(2)/rho2(1);
a_ = cos(deg2rad(theta(1)-theta(2))/2);
b_ = sin(deg2rad(theta(1)-theta(2))/2);
equiv_plane_angle = pi/2-deg2rad(theta(1)/2)-...
    acos((((r_*a_-1)/(r_*b_))^2+1)^-0.5);
n_equiv = [cos(equiv_plane_angle);sin(equiv_plane_angle)];
nhat = [nhat,R(90)*n_equiv,R(-90)*[-n_equiv(1);n_equiv(2)]];
for i = 1:size(nhat,2)
    % Compute augmented dynamical system ("constant input")
    Ab = [-A',-D;zeros(2,n+2)];
    Bb = zeros(n+2,1);
    Cb = [(B*nhat(:,i))',zeros(1,2)];
    Db = 0;
    V = weak_unobsv_sub(Ab,Bb,Cb,Db);
    % ensure it's just a single basis vector
    assert(size(V,2)==1,'weakly unobserable subspace dimension >1')
    % ensure that grad_r_l is not aligned with the basis vector "constant
    % input" augmented part
    w = V(n+1:end);
    for j = 1:size(grad_r_l,2)
        for sign1 = [-1,1]
            for sign2 = [-1,1]
                grad = [sign1*grad_r_l(1,j);sign1*grad_r_l(2,j)];
                if acos(grad.'*w/(norm(grad)*norm(w)))<align_tol
                    error(['weakly unobservable mode detected for nhat #%d, '...
                           'grad_r_l #%d, sign1=%d, sign2=%d'],i,j,sign1,sign2);
                end
            end
        end
    end
end
fprintf('Conditions 2 and 3 hold as as long as r_1(t)~=0 a.e. [0,t_f]\n');

% NEW
% Check that constant projection does not happen
D = [eye(2);zeros(2)];
% angles = linspace(-theta(1)/2,theta(1)/2,1e3);
% for i = 1:numel(angles)
%     % Acceleration pointing direction
%     ang = angles(i);
%     phat = R(ang)*ey;
%     % Compute augmented system ("constant input")
%     Ab = [-A',-D;zeros(2,n+2)];
%     Bb = zeros(n+2,1);
%     Cb = nan(n-1,n+2);
%     for j = 1:size(Cb,1)
%         Cb(j,:) = [((-A)^j*B*phat)',zeros(1,2)];
%     end
%     Db = zeros(size(Cb,1),1);
%     V = weak_unobsv_sub(Ab,Bb,Cb,Db);
%     disp(V)
% end

Ab = [-A',D;zeros(2,n+2)];
Cb = [B',zeros(2,2)];
[T,L] = eig(Ab);

% Simulate the system
dynamics = @(t,x) Ab*x;
adj0 = real(T(:,1)); %[0.5;0.5;0;0;0;0]
[t,adj] = ode45(dynamics,[0,10],adj0);
adj = adj';
y = Cb*adj;

% figure(1);
% clf()
% grid on
% hold on
% axis equal
% plot(y(1,:),y(2,:),'color','black','marker','x');
% plot(y(1,1),y(2,1),'color','red','marker','.','markersize',20);

ang = -10;
dir = R(ang)*ey;
Cb2 = dir'*Cb*Ab;
V_unobsv = weak_unobsv_sub(Ab,zeros(n+2,1),Cb2,0);

% Simulate the system
dynamics = @(t,x) Ab*x;
coeff = (Cb*V_unobsv)\(R(ang-20)*ey);
adj0 = coeff(1)*V_unobsv(:,1)+coeff(2)*V_unobsv(:,2); %[0.5;0.5;0;0;0;0]
[t,adj] = ode45(dynamics,[0,10],adj0);
adj = adj';
ydot_along_facet = Cb2*adj;
y = Cb*adj;

figure(1);
clf()
grid on
hold on
axis equal
plot(y(1,:),y(2,:),'color','black','marker','x');
plot(y(1,1),y(2,1),'color','red','marker','.','markersize',20);

%% Local functions

function X = preimage(A,V)
% PREIMAGE Finds X such that im(X)={x: A*x \in im(V)}=A^(-1)*im(V).
% Note that A neet not be invertible. Uses the identity:
%
%   A^(-1)*im(V) = ker[(A'*ker[V'])']
%
% Syntax:
%
% X = preimage(A,V)
%
% Inputs:
%
% A [double(n,m)] : matrix to use for the preimage.
% V [double(n,r)] : matrix whose image (i.e. range) defines the subspace
% whose preimage by A is to be taken.
% 
% Outputs:
%
% X [double(m,q)] : matrix whose range is the desired preimage.
X = null((A'*null(V'))');
end

function W = sum_sub(A,B)
% SUM_SUB Sum of subspaces such that im(W)=im(A)+im(B)
%
% Syntax:
%
% W = sum_sub(A,B)
%
% Inputs:
%
% A [double(n,m)] : first subspace im(A).
% B [double(n,r)] : second subspace im(B).
% 
% Outputs:
%
% W [double(n,q)] : subspace sum im(A)+im(B).
ra = rank(A);
rb = rank(B);
[Ua,~,~] = svd(A);
[Ub,~,~] = svd(B);
T =[Ua(:,1:ra),Ub(:,1:rb)];
W = im(T);
end

function V = im(A)
% IM Computes V such that im(V)=im(A), V is one-to-one and V'*V=I.
%
% Syntax:
%
% V = im(A)
%
% Inputs:
%
% A [double(n,m)] : matrix whose image (i.e. range space) is to be taken.
% 
% Outputs:
%
% V [double(n,r)] : one-to-one matrix such that im(V)=im(A) and V'*V=I.
[U,~,~] = svd(A);
r = rank(A);
n = size(U,1);
if r==n
    V = eye(r);
elseif r>0
    V = U(:,1:r);
else
    V = zeros(n,0);
end
end

function Vs = weak_unobsv_sub(A,B,C,D)
% WEAK_UNOBSV_SUB Find the weakly unobservable subspace for the linear system
% \Sigma = (A,B,C,D). Based on Trentelman et al. page 162.
%
% Original author: Matt Harris, April 16, 2013
%
% Syntax:
%
% Vs = weak_unobsv_sub(A,B,C,D)
%
% Inputs:
%
% A [double(n,n)] : state-space A matrix.
% B [double(n,m)] : state-space B matrix.
% C [double(p,n)] : state-space C matrix.
% D [double(p,m)] : state-space D matrix.
% 
% Outputs:
%
% Vs [double(n,r)] : matrix whose columns span the weakly ubonservable subspace
%                    \Sigma.
n = size(A,1);
m = size(B,2);
p = size(C,1);
AC = [A;C];
BD = [B;D];
V = eye(n,n);
for i = 1:n
    q  = size(V,2);
    S1 = [V, zeros(n,m); zeros(p,q), zeros(p,m)];
    S2 = sum_sub(S1,BD);
    V  = preimage(AC,S2);
    % if isempty(V)
    %     disp('none');
    % else
    %     disp(V)
    % end
end
Vs = V;
end

function Vs = strong_ctrbl_sub(A,B,C,D)
% STRONG_CTRBL_SUB Find the strongly controllable subspace for the linear system
% \Sigma = (A,B,C,D). Based on Trentelman et al. page 162.
%
% Syntax:
%
% Vs = strong_ctrbl_sub(A,B,C,D)
%
% Inputs:
%
% A [double(n,n)] : state-space A matrix.
% B [double(n,m)] : state-space B matrix.
% C [double(p,n)] : state-space C matrix.
% D [double(p,m)] : state-space D matrix.
% 
% Outputs:
%
% Vs [double(n,r)] : matrix whose columns span the strongly controllable
%                    subspace \Sigma.
Vs = weak_unobsv_sub(A',C',B',D');
Vs = null(Vs');
end
