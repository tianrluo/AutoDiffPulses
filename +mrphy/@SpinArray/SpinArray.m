% object for MRI simulations.
%
% Brief intro on methods:
%
%   Tianrui Luo, 2017
%}

classdef SpinArray < matlab.mixin.SetGet & matlab.mixin.Copyable
  methods (Static) % for quickly retriving attributes names. zjdsfyzs
    function n = immutable(), n={'dim', 'mask', 'nM'}; end
    function n = compact(),   n={'M_', 'T1_', 'T2_', 'gam_'}; end
    function n = dependent(), n={'M', 'T1', 'T2', 'gam'}; end
  end

  properties (SetAccess = immutable)
    dim;  % [nx, [ny, [nz, ...]]], i.e., Nd
    mask; % (*Nd) a.u., simulation mask,
    nM;
  end
  properties (SetAccess = public, GetAccess = public)
    M_   % (nM, xyz) spin vector, rotating frame
    T1_  % (nM,) Sec, dflt for grey matter
    T2_  % (nM,) Sec, dflt for grey matter
    gam_ % (nM,) Hz/G
  end
  properties (Dependent = true)
    M
    T1
    T2
    gam
  end

  methods (Access = public)
    function obj = SpinArray(dim, varargin)
      %INPUTS:
      % - dim (nx, (ny, (nz, ...)))
      %OPTIONALS
      % - mask (*Nd) logical
      % - M (*Nd, xyz)
      % - T1 (1,), global; (*Nd, ), spin-wise
      % - T2 (1,), global; (*Nd, ), spin-wise
      % - gam (1,), global; (*Nd, ), spin-wise
      import attr.*

      obj.dim = dim;

      %% dflts
      kv_c = [{'mask'}, obj.dependent; cell(1, 1+numel(obj.dependent))];
      arg = struct(kv_c{:});
      arg.mask = true(obj.dim);
      arg.T1  = ones(obj.dim)*1.47;
      arg.T2  = ones(obj.dim)*0.07;
      arg.gam = ones(obj.dim)*mrphy.utils.envMR('get','gam');
      arg.M   = cat(numel(obj.dim)+1, zeros([obj.dim, 2]), ones(obj.dim));

      arg = attrParser(arg, varargin);

      %%
      [obj.mask, obj.nM] = deal(arg.mask, nnz(arg.mask));
      obj.T1_  = obj.extract(arg.T1);
      obj.T2_  = obj.extract(arg.T2);
      obj.gam_ = obj.extract(arg.gam);
      obj.M_   = obj.extract(arg.M);
    end

    function [Mo_, Mhst_] = applypulse(obj, pulse, varargin)
      %INPUTS:
      % - pulse (1,) @Pulse
      %OPTIONALS:
      % - loc_   ^ loc   (nM, xyz)       ^ (*Nd, xyz), XOR
      % - b0Map_ | b0Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doCim [T/f]
      % - doEmbed [t/F]
      % - doUpdate [t/F]
      %OUTPUTS:
      % - Mo_   (nM, xyz)     | (*Nd, xyz)
      % - Mhst_ (nM, xyz, nT) | (*Nd, xyz, nT), evolving history
      import attr.*

      %% parsing
      [arg.loc,   arg.loc_] = deal([], []);
      [arg.b0Map, arg.b0Map_] = deal([], []);
      [arg.b1Map, arg.b1Map_] = deal([], []);
      [arg.doCim, arg.doEmbed, arg.doUpdate] = deal(true, false, false);

      arg = attrParser(arg, varargin);

      kw = {  'loc',arg.loc,     'loc_',arg.loc_ ...
            , 'b0Map',arg.b0Map, 'b0Map_',arg.b0Map_ ...
            , 'b1Map',arg.b1Map, 'b1Map_',arg.b1Map_ ...
            , 'doEmbed',false};
      beff_ = obj.pulse2beff(pulse, kw{:});

      [Mo_, Mhst_] = mrphy.sims.blochsim(obj.M_, beff_, obj.T1_, obj.T2_ ...
                                         , pulse.dt, obj.gam_ ...
                                         , arg.doCim);

      if arg.doUpdate, obj.M_ = Mo_; end
      if arg.doEmbed
        [Mo_, Mhst_] = deal(obj.embed(Mo_), obj.embed(Mhst_));
      end
    end
    
    function [M_ss, Mhst_] = applypulse_ss(obj, pulse, varargin)
      %Comupte steady state Magnetization
      %INPUTS:
      % - pulse (1,) @Pulse
      %OPTIONALS:
      % - loc_   ^ loc   (nM, xyz)       ^ (*Nd, xyz), XOR
      % - b0Map_ | b0Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doCim [T/f]
      % - doEmbed [t/F]
      % - doUpdate [t/F]
      %OUTPUTS:
      % - Mo_   (nM, xyz)     | (*Nd, xyz)
      % - Mhst_ (nM, xyz, nT) | (*Nd, xyz, nT), evolving history
      import attr.*

      %% parsing
      [arg.loc,   arg.loc_] = deal([], []);
      [arg.b0Map, arg.b0Map_] = deal([], []);
      [arg.b1Map, arg.b1Map_] = deal([], []);
      [arg.doCim, arg.doEmbed, arg.doUpdate] = deal(true, false, false);

      arg = attrParser(arg, varargin);

      kw = {  'loc',arg.loc,     'loc_',arg.loc_ ...
            , 'b0Map',arg.b0Map, 'b0Map_',arg.b0Map_ ...
            , 'b1Map',arg.b1Map, 'b1Map_',arg.b1Map_ ...
            , 'doEmbed',false};
      beff_ = obj.pulse2beff(pulse, kw{:});
      
      %[Mo_, Mhst_] = mrphy.sims.blochsim(obj.M, beff_, obj.T1_, obj.T2_ ...
      [Mo_, Mhst_] = mrphy.sims.blochsim(obj.M_, beff_, obj.T1_, obj.T2_ ...
                                         , pulse.dt, obj.gam_ ...
                                         , arg.doCim);
      Mo_;
      Tr=55; %ms
      T1_c=1300; %ms
      E1=exp(-Tr/T1_c);
      alpha=15;
      
      
      Mo_(isnan(Mo_))=0; %nan to 0
      
      if length(size(Mo_))==4
          beta=atan((Mo_(:,:,:,1).^2+Mo_(:,:,:,2).^2).^0.5./Mo_(:,:,:,3));
      elseif length(size(Mo_))==2
          beta=atan((Mo_(:,1).^2+Mo_(:,2).^2).^0.5./Mo_(:,3));
      end
      beta;
      M_ss=zeros(size(Mo_));
      if length(size(Mo_))==4
          M_ss(:,:,:,3)=(1-E1)./(1-cos(beta).*cosd(alpha).*E1);
      elseif length(size(Mo_))==2
          M_ss(:,3)=(1-E1)./(1-cos(beta).*cosd(alpha).*E1);
      end
      %M_ss(:,:,:,1)=M_ss(:,:,:,3).*tan(beta);
      
      if arg.doUpdate, obj.M_ = M_ss; end
      if arg.doEmbed
        [M_ss, Mhst_] = deal(obj.embed(M_ss), obj.embed(Mhst_));
      end
    end

    function st = asstruct(obj)
      warning('off', 'MATLAB:structOnObject')
      st = struct(obj);
      warning('on', 'MATLAB:structOnObject')
    end

    function Mr_ = freeprec(obj, dur, varargin)
      %INPUTS:
      % - dur (1,) Sec, duration of free precision
      %OPTIONALS:
      % - b0Map_ | b0Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils), "Hz"
      % - doEmbed [t/F]
      % - doUpdate [t/F]
      %OUTPUTS:
      % - Mr_ (nM, xyz) | (*Nd, xyz)
      import attr.*

      %% parsing
      [arg.b0Map, arg.b0Map_] = deal([], []);
      [arg.doEmbed, arg.doUpdate] = deal(false, false);

      arg = attrParser(arg, varargin);

      assert( isempty(arg.b0Map) || isempty(arg.b0Map_) ) % Not both
      if ~isempty(arg.b0Map), arg.b0Map_ = obj.extract(arg.b0Map); end

      Mr_ = freePrec(obj.M_, dur, obj.T1_, obj.T2_, arg.b0Map_);

      if arg.doEmbed, Mr_ = obj.embed(Mr_); end
      if arg.doUpdate, obj.M_ = Mr_; end
    end

    function beff_ = pulse2beff(obj, pulse, varargin)
      %INPUTS:
      % - pulse (1,) mrphy.@Pulse
      %OPTIONALS:
      % - loc_   ^ loc   (nM, xyz)       ^ (*Nd, xyz), XOR
      % - b0Map_ | b0Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - b1Map_ | b1Map (nM, 1, nCoils) ^ (*Nd, 1, nCoils)
      % - doEmbed [t/F]
      %OUTPUTS:
      % - beff_ (nM, xyz, nT) | (*Nd, xyz, nT)
      import attr.*

      %% parsing
      [arg.loc,   arg.loc_] = deal([], []);
      [arg.b0Map, arg.b0Map_] = deal([], []);
      [arg.b1Map, arg.b1Map_] = deal([], []);
      arg.doEmbed = false;

      arg = attrParser(arg, varargin);

      assert( isempty(arg.loc) ~= isempty(arg.loc_) ) % XOR
      if isempty(arg.loc_), arg.loc_ = obj.extract(arg.loc); end

      assert( isempty(arg.b0Map) || isempty(arg.b0Map_) ) % Not both
      if ~isempty(arg.b0Map), arg.b0Map_ = obj.extract(arg.b0Map); end

      assert( isempty(arg.b1Map) || isempty(arg.b1Map_) ) % Not both
      if ~isempty(arg.b1Map), arg.b1Map_ = obj.extract(arg.b1Map); end

      %%
      kw = {'gam', obj.gam_, 'b0Map', arg.b0Map_, 'b1Map', arg.b1Map_};
      beff_ = pulse.beff(arg.loc_, kw{:});
      if arg.doEmbed, beff_ = obj.embed(beff_); end
    end

  end
  
  methods % Utilities
    function v_ = extract(obj, v)
      [mask_t, ndim] = deal(obj.mask, numel(obj.dim));
      s_v = size(v);
      if numel(s_v) < ndim, s_v = [s_v, ones(1, ndim-numel(s_v))]; end
      shape_v = [s_v, 1]; % [*Nd, ..., 1]
      v = reshape(v, prod(obj.dim), []);
      v_ = reshape(v(mask_t,:),[obj.nM,shape_v(ndim+1:end)]); % (nM, ...)
    end

    function v = embed(obj, v_)
      shape_v_ = [size(v_), 1];
      v = nan([prod(obj.dim), shape_v_(2:end)]);
      v(obj.mask,:) = v_(:,:);
      v = reshape(v, [obj.dim, shape_v_(2:end)]);
    end
  end

  methods % set and get, sealed if the property cannot be redefined
    %% Dependent variables
    % WARNING
    % DO NOT proceed indexed/masked assignment to non-compact properties.
    function set.T1(obj, v),  obj.T1_  = obj.extract(v); end
    function set.T2(obj, v),  obj.T2_  = obj.extract(v); end
    function set.gam(obj, v), obj.gam_ = obj.extract(v); end
    function set.M(obj, v),   obj.M_   = obj.extract(v); end

    function v = get.T1(obj),  v = obj.embed(obj.T1_);  end
    function v = get.T2(obj),  v = obj.embed(obj.T2_);  end
    function v = get.gam(obj), v = obj.embed(obj.gam_); end
    function v = get.M(obj),   v = obj.embed(obj.M_);   end
  end
end
