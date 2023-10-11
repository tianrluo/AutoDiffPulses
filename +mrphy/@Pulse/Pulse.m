% object of MRI excitation pulse
%
% Brief intro on methods:
%
%   Tianrui Luo, 2019
%}

classdef Pulse < matlab.mixin.SetGet & matlab.mixin.Copyable
  properties (SetAccess = immutable)
    dt = mrphy.utils.envMR('get', 'dt');       % s
    gmax = mrphy.utils.envMR('get', 'gmax');   % G/cm
    smax = mrphy.utils.envMR('get', 'smax');   % G/cm/s
    rfmax = mrphy.utils.envMR('get', 'rfmax'); % G
  end
  properties (SetAccess = public, GetAccess = public)
    rf % (1, nT, (nCoils)), G
    gr % (xyz, nT), G/cm
    desc = 'generic pulse'
  end
  
  methods (Access = public)
    function obj = Pulse(varargin)
      % The constructor of a class will automatically create a default obj.
      import attr.*

      st = cutattrs(struct(varargin{:}), {}, properties('mrphy.Pulse'));
      for fName = fieldnames(st)', obj.(fName{1}) = st.(fName{1}); end
      assert(size(obj.rf,1)==1 && size(obj.gr,1)==3);
      assert(size(obj.rf,2)==size(obj.gr,2));
      % TODO:
      % - add sanity checks
    end

    function p_n = interpT(obj, dt_n, method)
      if abs(dt_n - obj.dt)/obj.dt < 1e-6, p_n = copy(obj); return; end
      % _o: old; _n: new
      if ~exist('method', 'var'), method = 'linear'; end
      method = lower(method);
      if ~strcmpi(method, 'linear')
        warning('Careful: non-linear interp can violate `gmax/smax/rfmax`');
      end

      [dt_o, nT] = deal(obj.dt, size(obj.rf, 2));

      t_o = ((0:nT)*dt_o)'; % insert 0 at the beginning
      fn_0 = @(x)[zeros(size(x(:,1,:))), x]; % pad 0 at the beginning along nT

      t_n = (dt_n:dt_n:(dt_o*nT))';

      rf_n = shiftdim(interp1(t_o, shiftdim(fn_0(obj.rf),1), t_n, method), -1);
      gr_n = interp1(t_o, fn_0(obj.gr).', t_n, method).';

      p_n = mrphy.Pulse('rf',rf_n, 'gr',gr_n, 'dt',dt_n ...
                        , 'gmax',obj.gmax, 'smax',obj.smax, 'rfmax',obj.rfmax);
      p_n.desc = [obj.desc, ' interpT''ed: dt = ', num2str(dt_n)];
    end

    function beff = beff(obj, loc, varargin)
      %INPUTS:
      % - loc (*Nd, xyz)
      %OPTIONALS:
      % - b0Map (*Nd)
      % - b1Map (*Nd, 1, nCoils)
      % - gam (1,)
      %OUTPUTS:
      % - beff (*Nd, xyz, nT)
      import attr.*

      [arg.b0Map,arg.b1Map,arg.gam] =deal([],[],mrphy.utils.envMR('get','gam'));
      arg = attrParser(arg, varargin);
      kw = [fields(arg), struct2cell(arg)]';
      beff = mrphy.beffective.rfgr2beff(obj.rf, obj.gr, loc, kw{:});
    end

    function st = asstruct(obj)
      warning('off', 'MATLAB:structOnObject')
      st = struct(obj);
      warning('on', 'MATLAB:structOnObject')
    end

  end
    
  methods % set and get, sealed if the property cannot be redefined
    % TODO:
    % - add sanity checks
  end

end
