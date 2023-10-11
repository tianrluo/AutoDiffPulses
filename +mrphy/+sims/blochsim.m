function [Mo, Mhst] = blochsim(Mi, Beff, T1, T2, dt, gam, doCim)
% function [Mo, Mhst] = blochsim(Mi, Beff, T1, T2, dt, gam, doCim)
% bloch Simulation for multiple spin with globally or spin-wisely prescripted
% parameters and pulse
%
%INPUTS:
% - Mi (*Nd, xyz): input spins' magnetizations
% - Beff (*Nd, xyz, nT), Gauss, spin-wise;
% - T1 & T2, Sec: globally or spin-wisely defined T1 and T2
%     (1,), global
%     (*Nd, 1), spin-wise
% - dt (1,) Sec: temporal simulation step size.
% - gam, Hz/G, gyro frequency ratio:
%     (1,), global
%     (*Nd, 1), spin-wise
% - doCim [t/F], call c-based simulation
% - doGPU [t/F], use matlab GPU features
%OUTPUTS:
% If doGPU is enabled, the OUTPUTs will be as gpuArray
% - Mo   (*Nd, xyz, nTs): output spins' magnetizations
% - Mhst (*Nd, xyz, nTs): output spins' magnetizations history
%Notices:
% 1. Not much sanity check inside this function, user is responsible for
%    matching up the dimensions.
% 2. For forking, user may want to avoid using cross() in MatLab, which involves
%    many permute()-calls and is time-consuming
% 3. Put decays at the end of each time step may still be problematic, since
%    physically the spin decays continuously, this noise/nuance may worth study
%    for applications like fingerprinting simulations, etc.
import mrphy.*

if nargin == 0, test(); return; end

[dt0, gam0] = utils.envMR('get', 'dt', 'gam');
if ~exist('dt',    'var') || isempty(dt),    dt    = dt0;   end % Sec
if ~exist('gam',   'var') || isempty(gam),   gam   = gam0;  end % Hz/G
if ~exist('doCim', 'var') || isempty(doCim), doCim = false; end

shape = size(Mi); % (*Nd, xyz)
shape = shape(1:end-1); % (*Nd)

doHist = (nargout > 1);

Mi = reshape(Mi, [], 3);
Beff = reshape(Beff, [], 3, size(Beff, numel(shape)+2)); % (prod(*Nd), xyz, nT)
[T1, T2, gam] = deal(T1(:), T2(:), gam(:));

nT = size(Beff, 3);

if doCim
  % harder to deal w/ dim in C, adjust dims in matlab before pass in C
  if doHist, [Mo, Mhst] = mrphy.sims.blochcim(Mi, double(Beff), T1,T2,dt,gam);
  else,      Mo = mrphy.sims.blochcim(Mi, double(Beff), T1, T2, dt, gam);
  end
  Mo = reshape(Mo, [shape, 3]);
  if doHist, Mhst = reshape(Mhst, [shape, 3, size(Mhst,3)]); end
  return;
end

[E1, E2, gambar] = deal(exp(-dt./T1), exp(-dt./T2), (2*pi*gam));

% negative to correct cross product (x-prod) direction;
Bmag = sqrt(sum(Beff.*Beff, 2)); % self multiplication is faster than power
niBmag = -1./Bmag;
niBmag(isinf(niBmag)) = 0;
Bn   = bsxfun(@times, Beff, niBmag); % Bn, normalized Beff;

phi = bsxfun(@times, Bmag, gambar)*dt;
Cp = cos(phi); % (*Nd, 1, nTs) or (1, 1, nTs)
Cp_1 = 1 - Cp;   % trick for Rotation Matrix
Sp = sin(phi); % (*Nd, 1, nTs) or (1, 1, nTs)

SpBn   = bsxfun(@times, Bn, Sp);   % trick for Rotation Matrix
CpBn_1 = bsxfun(@times, Bn, Cp_1); % trick for Rotation Matrix

[Mx0, My0, Mz0] = deal(Mi(:,1), Mi(:,2), Mi(:,3));
if doHist, [Mox, Moy, Moz] = deal(zeros(prod(shape), nT)); end

doFlag = any(niBmag ~= 0, 1);

% Updates are calculated with rotation-then-decay, instead of the canonical
% differential equation expression.
% Discretizing differential equation may cause precision issue.

% full lower-case variables are local in loop, o.w. not local
for istep = 1:nT
  if doFlag(istep)
    % step-wisely extract pre-processed variables
    bn     = Bn(:,:,istep);
    spbn   = SpBn(:,:,istep);
    cpbn_1 = CpBn_1(:,:,istep);

    ip = sum(bsxfun(@times, bn, [Mx0, My0, Mz0]), 2); % vector inner product
    cp = Cp(:, :, istep);

    % explicitly express cross(bn, Mo_ii_1) as a matrix vector multiplication
    mx1 =  cp       .*Mx0 -spbn(:,3).*My0 +spbn(:,2).*Mz0 +ip.*cpbn_1(:,1);
    my1 =  spbn(:,3).*Mx0 +cp       .*My0 -spbn(:,1).*Mz0 +ip.*cpbn_1(:,2);
    mz1 = -spbn(:,2).*Mx0 +spbn(:,1).*My0 +cp       .*Mz0 +ip.*cpbn_1(:,3);
  else
    mx1 = Mx0;
    my1 = My0;
    mz1 = Mz0;
  end
  % relaxation effects: "1" in Mz0 since M0=1 by assumption
  % also, update Mo_ii_1;
  Mx0 = mx1.*E2;
  My0 = my1.*E2;
  Mz0 = mz1.*E1 + 1-E1;
  if doHist, [Mox(:,istep),Moy(:,istep),Moz(:,istep)] = deal(Mx0,My0,Mz0); end
end
Mo = reshape([Mx0, My0, Mz0], [shape, 3]);
if doHist
  Mhst = reshape(permute(cat(3, Mox, Moy, Moz), [1,3,2]), [shape, 3, nT]);
end

end

%%
function test()
prefix = mfilename('fullpath');
disp('------------------------');
disp([prefix, '.test()']);

Mi = [1,0,0; 0,1,0; 0,0,1];
nt = 512;
t = (0:nt-1)';
Beff = 10 * [cos(t/nt * 2*pi), sin(t/nt * 2*pi), atan(t-round(nt/2))/pi];
Beff = permute(Beff, [3,2,1]);

%% test -- 1
Mo_S = mrphy.sims.blochsim(Mi, Beff, 1, 4e-2, [],[], false);
tmp = Mo_S - [ 0.559535641648385,  0.663342640621335,  0.416341441715101;
               0.391994737048090,  0.210182892388552, -0.860954821972489;
              -0.677062008711222,  0.673391604920576, -0.143262993311057];


assert(norm(tmp(:)./Mo_S(:))<=1e-9);
[~, Mhst] = mrphy.sims.blochsim(Mi, Beff, 1, 4e-2, [],[], false);
tmp = Mo_S - Mhst(:,:,end);
assert(norm(tmp(:)./Mo_S(:))<=1e-9);

Mo_C = mrphy.sims.blochsim(Mi, Beff, 1, 4e-2, [],[], true);
[~, Mhst] = mrphy.sims.blochsim(Mi, Beff, 1, 4e-2, [],[], true);
tmp = Mo_C - Mhst(:,:,end);
assert(norm(tmp(:)./Mo_C(:))<=1e-9);
tmp = Mo_S - Mo_C;
assert(norm(tmp(:)./Mo_S(:))<=1e-9);

%%
disp([prefix, '.test() done']);
% keyboard;
end
