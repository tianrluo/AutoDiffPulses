function beff = rfgr2beff(rf, gr, loc, varargin)
%INPUTS:
% - rf (1, nT, (nCoils))
% - gr (xyz, nT)
% - loc (*Nd, xyz)
%OPTIONALS:
% - b0Map (*Nd)
% - b1Map (*Nd, 1, nCoils)
% - gam (1,)
%OUTPUTS:
% - beff (*Nd, xyz, nT)
import attr.*

shape = size(loc);
nT = size(rf, 2);
shape = shape(1:end-1);
loc = reshape(loc, [], 3);  % -> (prod(Nd), xyz)

%% parsing
[arg.b0Map, arg.b1Map, arg.gam] = deal([], [], mrphy.utils.envMR('get', 'gam'));
arg = attrParser(arg, varargin);

[b0Map, b1Map, gam] = getattrs(arg, {'b0Map', 'b1Map', 'gam'});

%% form beff
bz = reshape(loc*gr, [shape, 1, nT]); % (prod(Nd), nT) -> (*Nd, 1, nT)
if ~isempty(b0Map), bz = bsxfun(@plus, bz, bsxfun(@rdivide, b0Map, gam)); end

rf = repmat(rf, prod(shape), 1); % -> (prod(Nd), nT, (nCoils))
if ~isempty(b1Map), rf=bsxfun(@times, reshape(b1Map,prod(shape),1,[]), rf); end
rf = reshape(sum(rf, 3), [shape, 1, nT]); % (prod(Nd),nT,(nCoils)) -> (*Nd,1,nT)

%% return
beff = cat(numel(shape)+1, real(rf), imag(rf), bz); % (*Nd, xyz, nT)

end
