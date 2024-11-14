function w=RIM(n,p)

% =========================================================================
% Regular increasing monotone (RIM) quantifier based weights generation
% =========================================================================

w=zeros(1,n);

for i=1:n
    w(i)=(i/n).^p-((i-1)/n)^p;
end
end