% 3D object for MRI excitation simulations.
%{
% 
% See Also: mrphy.SpinArray
%
% Brief intro on methods:
%
%   Tianrui Luo, 2017
%}

classdef SpinCube < mrphy.SpinArray
  methods (Static) % for quickly retriving attributes names. zjdsfyzs
    function n = compact(), n=[{'loc_','b0Map_'}, compact@SpinArray()]; end
    function n = dependent()
      n=[{'loc','b0Map','ofstn','res'}, dependent@mrphy.SpinArray()];
    end
  end
  properties (SetAccess = public, GetAccess = public)
    fov;    % (1, xyz,) cm, field of view
    ofst;   % (1, xyz,) cm, offset of the fov;
    b0Map_; % (nM,)  Hz
  end
  properties (SetAccess = protected, GetAccess = public)
    loc_;   % (nM, xyz) cm
  end
  properties (Dependent)
    loc;   % (*Nd, xyz) cm
    b0Map; % (*Nd, 1) Hz

    ofstn; % (xyz,) [-0.5, 0.5], ofst./fov
    res;   % (xyz,) cm, resolution
  end
  
  methods (Access = public)
    function obj = SpinCube(fov, dim, ofst, varargin)
      %{
      %INPUTS:
        - fov  (1,xyz,) cm
        - dim  (1,xyz,)
        - ofst (1,xyz,) cm
      %OPTIONALS
        - M     (*Nd, xyz)
        - gam   (1,), global; (*Nd, 1), spin-wise
        - T1    (1,), global; (*Nd, 1), spin-wise
        - T2    (1,), global; (*Nd, 1), spin-wise
        - mask  (*Nd) logical
        - b0Map (*Nd) Hz
      %}
      import attr.*
      % defaults
      [arg.b0Map, arg.b0Map_] = deal([], []);

      [arg, extra] = attrParser(arg, varargin);
      obj@mrphy.SpinArray(dim, extra{:});

      assert( isempty(arg.b0Map) || isempty(arg.b0Map_) ) % Not both
      if ~isempty(arg.b0Map), arg.b0Map_ = obj.extract(arg.b0Map); end

      obj.b0Map_ = arg.b0Map_;
      obj.fov = fov;
      obj.ofst = ofst;

      obj.update_loc_(); % set obj.loc_
    end
    
    function [Mo, Mhst] = applypulse(obj, pulse, varargin)
      %INPUTS:
      % - pulse (1,) mrphy.Pulse
      %OPTIONALS:
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doCim [T/f]
      % - doEmbed [t/F]
      % - doUpdate [t/F]
      %OUTPUTS:
      % - Mo_   (nM, xyz)     | (*Nd, xyz)
      % - Mhst_ (nM, xyz, nT) | (*Nd, xyz, nT), evolving history
      kw = [{'loc_',obj.loc_, 'b0Map_',obj.b0Map_}, varargin];
      [Mo, Mhst] = applypulse@mrphy.SpinArray(obj, pulse, kw{:});
    end
    
    function [Mo, Mhst] = applypulse_ss(obj, pulse, varargin)
      %INPUTS:
      % - pulse (1,) mrphy.Pulse
      %OPTIONALS:
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doCim [T/f]
      % - doEmbed [t/F]
      % - doUpdate [t/F]
      %OUTPUTS:
      % - Mo_   (nM, xyz)     | (*Nd, xyz)
      % - Mhst_ (nM, xyz, nT) | (*Nd, xyz, nT), evolving history
      kw = [{'loc_',obj.loc_, 'b0Map_',obj.b0Map_}, varargin];
      [Mo, Mhst] = applypulse_ss@mrphy.SpinArray(obj, pulse, kw{:});
    end

    function [xcoord, ycoord, zcoord] = get_coords(obj)
      cSub = mrphy.utils.ctrSub(obj.dim);
      coords_c = arrayfun(@(d, c, f, o)((1:d)-c)/d*f + o, ... % cm
                          obj.dim, cSub, obj.fov, obj.ofst, 'Uni', false);
      [xcoord, ycoord, zcoord] = deal(coords_c{:});
    end

    function beff = pulse2beff(obj, pulse, varargin)
      %INPUTS:
      % - pulse (1,) mrphy.Pulse
      %OPTIONALS:
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doEmbed [t/F]
      %OUTPUTS:
      % - beff_ (nM, xyz, nT) | (*Nd, xyz, nT)
      kw = [{'loc_',obj.loc_, 'b0Map_',obj.b0Map_}, varargin];
      beff = pulse2beff@mrphy.SpinArray(obj, pulse, kw{:});
    end
  end % of methods (Access = public)
  
  methods % set & get methods
    function set.fov(obj, v)
      assert( numel(v) == 3 )
      obj.fov = v;
      obj.update_loc_();
    end

    function set.ofst(obj, v)
      assert( numel(v) == 3 )
      obj.ofst = v;
      obj.update_loc_();
    end

    function res = get.res(obj), res = obj.fov./obj.dim; end
    function ofstn = get.ofstn(obj)
      ofstn = obj.ofst./obj.fov;
      ofstn(isinf(ofstn) | isnan(ofstn)) = 0;
    end

    function set.b0Map(obj, v), obj.b0Map_ = obj.extract(v); end
    function v = get.b0Map(obj), v = obj.embed(obj.b0Map_); end

    function v = get.loc(obj), v = obj.embed(obj.loc_); end
  end

  methods (Access = protected)
    function update_loc_(obj)
      if isempty(obj.fov)||isempty(obj.ofst)||isempty(obj.dim), return; end
      [xcoord, ycoord, zcoord] = obj.get_coords();
      [xloc, yloc, zloc] = ndgrid(xcoord, ycoord, zcoord);
      % matlab does not output a 1D array when mask is all `true`
      [xloc, yloc, zloc] = deal(xloc(:), yloc(:), zloc(:));
      obj.loc_ = cat(2, xloc(obj.mask), yloc(obj.mask), zloc(obj.mask));
    end
  end
  
end

