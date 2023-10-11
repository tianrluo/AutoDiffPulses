function gr = k2g(TxRx, kpcm, dt, gam)
% INPUT
% - TxRx, str,
%     Tx, k is assumed to end at the 0, output gr(end,:) == 0
%     Rx, k is assumed to start at the 0, output gr(1,:) == 0
% - kpcm (ndim, nT, ...) cycle/cm
%OPTIONAL
% - dt,  (1,), Sec
% - gam, (1,), Hz/G
% OUTPUT
% - gr (ndim, nT, ...), G/cm
if nargin == 0, test(); return; end

[dt0, gam0] = mrphy.utils.envMR('get', 'dt','gam');
if ~exist('dt',  'var') || isempty(dt),  dt  = dt0;  end % Sec
if ~exist('gam', 'var') || isempty(gam), gam = gam0; end % Hz/Gauss

gr = reshape([kpcm(:,1,:), diff(kpcm(:,:,:),1,2)]/gam/dt, size(kpcm));

switch lower(TxRx)
  case 'tx', assert(all(reshape(kpcm(:,end,:),[],1)==0), 'Tx k must end at 0');
  case 'rx' % nothing
  otherwise, error('Transmit (tx) or Receive (rx)???');
end

end

function test()
prefix = mfilename('fullpath');
disp('------------------------');
disp([prefix, '.test()']);
[dt, gam, k] = deal(4e-6, 4.2576e3, [1, 2, 3, 4, 0]);
gr_t = mrphy.utils.k2g('tx', k, dt, gam);
gr_r = mrphy.utils.k2g('rx', k, dt, gam);

assert(max(reshape(abs(mrphy.utils.g2k('tx',gr_t,dt,gam)-k),[],1)<=1e-12));
assert(max(reshape(abs(mrphy.utils.g2k('rx',gr_r,dt,gam)-k),[],1)<=1e-12));
disp([prefix, '.test() passed']);
end
