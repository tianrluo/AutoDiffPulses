function g = s2g(s, dt)
% function g = s2g(s, dt)
% INPUT
%  s,  (ndim, nT, ...), G/cm/Sec
%  dt,  (1)  , optional, Sec
% OUTPUT
%  g,  (ndim, nT, ...), G/cm
if nargin == 0, test(); return; end

dt0 = mrphy.utils.envMR('get', 'dt');
if ~exist('dt', 'var') || isempty(dt),  dt = dt0;  end % Sec

g = cumsum(s, 2)*dt;

end

function test()
prefix = mfilename('fullpath');
disp('------------------------');
disp([prefix, '.test()']);
[dt, gr] = deal(4e-6, [1, 2, 3, 4, 0]);
s = mrphy.utils.g2s(gr, dt);

assert(max(reshape(abs(mrphy.utils.s2g(s,dt)-gr),[],1)<=1e-12));
disp([prefix, '.test() passed']);
end
