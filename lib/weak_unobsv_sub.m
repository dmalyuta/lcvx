function Vs = weak_unobsv_sub(A,B,C,D)
% WEAK_UNOBSV_SUB Find the weakly unobservable subspace for the linear system
% \Sigma = (A,B,C,D). Based on Trentelman et al. page 162.
%
% Original author: Matthew W. Harris, April 16, 2013
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
