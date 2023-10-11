function varargout = envMR(mod, varargin)
% contains environmental variables: gmax, smax, dt, gam
%INPUTS:
% - mod str, case insensitive, {'set', 'get', 'get_s', 'reset'}
%   'set':   Set environmental persistent variables;
%   'get':   Get the values of persistent variables, return them separately;
%   'get_s': Get the values of queried variables, retuned as a structure;
%   'reset': Resetting the persistent variables to default values.
%NOTE:
% Many pulse design functions have separate environmental variable inputs that
% allows env vars to differ from the 'persistent' system env vars here.
% For instance, it can happen when one wants to empirically ensure the designed
% pulse does not violate any system env var restrictions.
% This heuristic of staying within the system limits is often due to the algs
% used are penalized methods rather than constraint methods.
import attr.*

if nargin == 0, test(); return; end

persistent mrEnv_p % _p: persistent, good to put these vars at the beginning.
% {Sec, Hz/Gauss, Gauss/cm, Gauss/cm/Sec, Gauss}
fName_c = {'dt','gam','gmax','smax','rfmax'};
if isempty(mrEnv_p) || strcmpi(mod, 'reset')
  % GE MR750 setting used in fMRIlab UMich, `-1e-9` to avoid save as int in mat
  mrEnv_p = cell2struct({4e-6, 4257.6, 5-1e-9, 12e3-1e-9, 0.25}, fName_c, 2);
  if strcmpi(mod, 'reset'), disp('...mrEnv_p in envMR reset');
  else, disp('...mrEnv_p in envMR initialized');
  end
end

% for cell input, e.g. envMR('get', {'dt', 'gam'}), envMR('set',{'dt'}).
if numel(varargin)==1 && iscell(varargin{1}), varargin = varargin{1}; end

switch lower(mod)
  case {'set', 'reset'}
    if ~strcmpi(mod,'reset') && ~isempty(varargin)
      mrEnv_p = attrParser(mrEnv_p, varargin, false);
    end
    varargout = {mrEnv_p};
  case 'get'
    varargout = getattrs(mrEnv_p, varargin, false);
    assert(nargout==numel(varargin));
  case 'get_s'
    if isempty(varargin), varargout = {mrEnv_p};
    else, varargout = {cell2struct(getattrs(mrEnv_p,varargin,0), varargin, 2)};
    end
  otherwise,  error('Unsupported set_get query');
end

end

function test()
import attr.*

prefix = mfilename('fullpath');
disp('------------------------');
disp([prefix, '.test()']);
env1_s = mrphy.utils.envMR('get_s'); %_s: struct, back up current values
fName_c = {'dt','gam','gmax','smax','rfmax'}; % {s, Hz/Gs, Gs/cm, Gs/cm/s, Gs}
assert(isequal(fName_c, fieldnames(env1_s)'), 'env var name mismatch');

[dt, gam, gmax, smax, rfmax] = mrphy.utils.envMR('get', fName_c{:});
assert(isequal(getattrs(env1_s,fName_c),{dt,gam,gmax,smax,rfmax}),'get failed');

env0_s = mrphy.utils.envMR('reset'); % env2.dt~= env[01].dt
env2_s = mrphy.utils.envMR('set', 'dt',exp(env0_s.dt)+exp(dt));
assert(~isequal(env1_s,env2_s), 'set test failed');
assert(isequal(mrphy.utils.envMR('reset'), env0_s), 'reset test failed');

% restore from backup
mrphy.utils.envMR('set', env1_s);
disp([prefix, '.test() passed']);
end
