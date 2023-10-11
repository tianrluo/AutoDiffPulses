function s = g2s(gr, dt)
% Slew Rate from Gradient
% INPUT
% - gr (ndim, nT, ...) Gauss/cm
% - dt,  (1)  , optional, Sec
% OUTPUT
% - s (ndim, nT, ...), G/cm/Sec
if nargin == 0, test(); return; end

dt0 = mrphy.utils.envMR('get','dt');
if ~exist('dt', 'var') || isempty(dt), dt = dt0;  end % Sec

% gr = [gr; gr(end,:)];
% s = diff(gr, 1,1)/dt; % Gauss/cm/Sec

s = reshape([gr(:,1,:), diff(gr(:,:,:), 1,2)]/dt, size(gr)); % Gauss/cm/Sec

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
