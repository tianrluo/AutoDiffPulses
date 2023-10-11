function [pulse_o, optInfos] = arctanAD_ss(target, cube, pulse_i, varargin)
% This fn nastily relies on file save/read to pass variables to python calls.
% Blame Matlab/Python poor interoperability.
% This function is basically a python function wrapper.
% The python part requires `scipy`, `mrphy` and `torch` (with CUDA).
%INPUTS
% - target (1,) structure:
%   .d (*Nd, xyz)
%   .weight (*Nd)
% - cube mrphy.@SpinCube obj
% - pulse_i mrphy.@Pulse obj
%OPTIONAL
% - b1Map (*Nd, nCoils)
% - niter (1,), dflt 10, #iteration in autodiff
% - niter_rf (1,), dflt 2, #iteration for updating rf
% - niter_gr (1,), dflt 2, #iteration for updating gr
% - err_meth str:
%   'l2': least square excitation error metric accounting for all M_{x,y,z};
%   'l2xy': ordinary transversal least square excitation error metric;
%   'ml2xy': transversal magnitude least square excitation error metric;
%   'l2z': longitudinal least square.
% - pen_meth str:
%   'null': no regularization.
%   'l2': least square, RF power regularizer.
% - eta (1,), Tikhonov coefficient for regularization, i.e., λ.
%   keyword `lambda` is occupied in python, to be consistent, use η.
% - gpuID (1,) which GPU to run the pulse design. If `-1`, use CPU.
% - doRelax[T/f], allow spins to relax during simulation.
% - doClean [T/f], remove temporary files at finish.
% - fName dflt 'arctanAD', name of the temporary files.

import attr.*

% parse
arg.b1Map = [];
[arg.niter, arg.niter_gr, arg.niter_rf] = deal(8, 2, 2);
arg.err_meth = 'l2xy';
[arg.pen_meth, arg.eta] = deal('l2', 4);
arg.gpuID = 0;
[arg.doRelax, arg.doClean] = deal(true, true);
arg.fName = 'adpulses_opt_arctanAD';

arg = attrParser(arg, varargin);
disp(['err_meth: ', arg.err_meth])

[arg.err_meth, arg.pen_meth] = deal(lower(arg.err_meth), lower(arg.pen_meth));
assert(ismember(arg.err_meth, {'null', 'l2', 'l2xy', 'ml2xy', 'l2z'}))
assert(ismember(arg.pen_meth, {'null', 'l2'}))

[m2pName, p2mName] = deal([arg.fName, '_m2p.mat'], [arg.fName, '_p2m.mat']);
gpuID = arg.gpuID;

%% python call
if size(target.d, 4) ~= 3  % -> (*Nd, xyz)
  d = target.d;
  target.d = cat(4, real(d), imag(d), sqrt(1-abs(d).^2));
end

[cube_st, pulse_st] = deal(cube.asstruct(), pulse_i.asstruct());

% `'-v7'` for scipy.io compatibility
save(m2pName, '-v7', 'target', 'cube_st', 'pulse_st', 'arg')

[p, ~, ~] = fileparts(mfilename('fullpath')); % .m/.py must be in the same path.
pyfile = [p, '/arctanAD_ss.py'];


cmd = ['python ', pyfile, ' ', m2pName, ' ', p2mName, ' ', num2str(gpuID)];
Err = system(cmd);  % python call
if Err, error('python call failed!!!'); end

%% back to matlab
% $p2mName contains the following arguments:
% - pulse_o
% - optInfos

mfile = matfile(p2mName);

% may need some reshaping before assigning to mrphy.Pulse
[pulse_st, arg_p2m] = deal(mfile.pulse_st, mfile.arg);
optInfos = arg_p2m.optInfos;
pulse_st.rf = pulse_st.rf(1,:,:) + 1i*pulse_st.rf(2,:,:);
pulse_o = copy(pulse_i);
[pulse_o.rf, pulse_o.gr] = deal(double(pulse_st.rf), double(pulse_st.gr));

if arg.doClean, delete(m2pName, p2mName); end

end
