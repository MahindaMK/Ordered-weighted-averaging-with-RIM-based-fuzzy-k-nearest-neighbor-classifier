function h=owamatrix(A,w)

% =========================================================================
% Ordered weighted averaging vectors
% =========================================================================

for i=1:size(A,1)
    apu = A(i,:);
    h(i) = sum(sort(apu,2,'descend').*w);
end

