%function demo_ss()
% Compares steady-state-aware pulse design (OV90_ss) with regular design (OV90).
% Both pulses are evaluated under steady-state Bloch simulation and compared
% against the steady-state target.  
% Hypothesis: pAD_ss should be closer to target_ss than pAD.

IniVar = matfile('./IniVars.mat');
cube        = IniVar.cube;
pIni        = IniVar.pIni_OV90;
target_OV90 = IniVar.target_OV90;

TR    = 60e-3;  % s
alpha = 0;      % deg, tip-down flip angle

% Derive IV (box) and OV masks from the existing OV90 target:
%   IV: Mz target == 1  (0-deg, no excitation)
%   OV: the rest of the object support
msk     = logical(cube.mask);
iv_mask = (target_OV90.d(:,:,:,3) == 1) & msk;
ov_mask = msk & ~iv_mask;

% Steady-state target: 0-deg IV, 90-deg OV
[target_ss.d, target_ss.weight] = mrphy.steady_state.spgr.target_ovs( ...
    cube, 0, 90, iv_mask, ov_mask, 'TR', TR, 'alpha', alpha);

% --- Design pulses ---
[pAD_ss, ~] = OV90_ss(cube, target_ss, pIni, TR, alpha);
[pAD,    ~] = OV90(cube, target_OV90, pIni);

%%
% --- Evaluate both pulses under steady-state simulation ---
M1_ss = mrphy.steady_state.spgr.spgr_ovs(cube, pAD,    'TR', TR, 'alpha', alpha, 'doEmbed', true);  % regular design
M2_ss = mrphy.steady_state.spgr.spgr_ovs(cube, pAD_ss, 'TR', TR, 'alpha', alpha, 'doEmbed', true);  % ss design

% --- Evaluate both pulses under regular bloch simulation ---
M1 = cube.applypulse(pAD, 'doEmbed',true);  % regular design
M2 = cube.applypulse(pAD_ss, 'doEmbed',true);  % ss design

%%
% --- Compare SS Simulation---
plot_res(M1_ss,M2_ss,cube,target_ss,'z') %M2_ss is closer to target than M1_ss, specifically in IV

% --- Compare regular Bloch Simulation ---
plot_res(M1,M2,cube,target_OV90,'z') %M1 is closer to target than M2 for regular blochsim

% --- compare the pulse waveforms --
adpulses.utils.plotPulseComparison(pAD,pAD_ss,[],["Regular design","Steady-state design"])

% =========================================================================
function [pAD_ss, optInfo] = OV90_ss(cube, target_ss, pIni, TR, alpha)
  [pAD_ss, optInfo] = adpulses.opt.arctanAD_spgr(target_ss, cube, pIni, ...
      'err_meth', 'l2z', ...
      'TR',       TR,    ...
      'alpha',    alpha, ...
      'doClean', false, 'gpuID', 0);
end

function [pAD, optInfo] = OV90(cube, target, pIni)
  [pAD, optInfo] = adpulses.opt.arctanAD(target, cube, pIni, ...
      'err_meth', 'l2z', ...
      'doClean', false, 'gpuID', 0);
end

function plot_res(MT_1, MT_2, cube, target, xy_z)
% inputs:
%   MT_1: [nx,ny,nz,3]
%   MT_2: [nx,ny,nz,3]
  
  if strcmpi(xy_z, 'xy')
    % xy component and transversal MLS NRMSE
    fn_res = @(MT)MT(:,:,:,1) + 1i*MT(:,:,:,2);
    fn_tile = @(P)abs(tile3dto2d(P));
    cl_res = [0, 1];
  elseif strcmpi(xy_z, 'z')
    % z component and longitudinal LS NRMSE
    fn_res = @(MT)MT(:,:,:,3);
    fn_tile = @(P)tile3dto2d(P);
    cl_res = [-1, 1];
  else, error('Unknown xy_z type');
  end
  
  d = fn_res(target.d);
  nan_mask = ~logical(cube.mask);
  d(nan_mask) = 0;
 
  [MT_1, MT_2] = deal(fn_res(MT_1), fn_res(MT_2));

  MT_1(nan_mask) = 0;
  MT_2(nan_mask) = 0;

  % reshapes
  [d, MT_1, MT_2] = deal(fn_tile(d), fn_tile(MT_1), fn_tile(MT_2));
  
  % NRMSE
  in_mask = ~nan_mask;
  nrmse = @(m) norm(m(in_mask) - d(in_mask)) / norm(d(in_mask));
  e1 = nrmse(MT_1);
  e2 = nrmse(MT_2);

  figure
  subplot(131);
  imagesc(d); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title('target');
  
  subplot(132);
  imagesc(MT_1); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title(sprintf('Regular design\nNRMSE=%.3f',e1));
  
  subplot(133);
  imagesc(MT_2); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title(sprintf('Steady-state design\nNRMSE=%.3f',e2));
end

% =========================================================================
function P = tile3dto2d(P)
  [nx, ny, nz] = size(P);
  nz_rt = sqrt(nz);
  [nc, nr] = deal(ceil(nz_rt), floor(nz_rt));
  P = cat(3, P, zeros([nx, ny, nr*nc-nz]));
  P = reshape(P, nx, ny*nc, []);
  P = reshape(permute(P, [2, 1, 3]), ny*nc, nx*nr).';
end


% % =========================================================================
% function compare_ss(M1, M2, target_ss, cube)
% % Show |Mxy| for target_ss, pAD (regular), and pAD_ss (ss-design), and
% % report NRMSE for each.
% 
%   fn_res  = @(MT) MT(:,:,:,3); %Mz  %@(MT) MT(:,:,:,1) + 1i*MT(:,:,:,2);   % Mxy
%   fn_tile = @(P)  abs(tile3dto2d(P));
%   cl_res  = [0, 1];
% 
%   d  = fn_res(target_ss.d);
%   nan_mask = ~logical(cube.mask);
%   d(nan_mask) = 0;
% 
%   r1 = fn_res(M1);  r1(nan_mask) = 0;
%   r2 = fn_res(M2);  r2(nan_mask) = 0;
% 
%   in_mask = ~nan_mask;
%   nrmse = @(m) norm(m(in_mask) - d(in_mask)) / norm(d(in_mask));
%   e1 = nrmse(r1);
%   e2 = nrmse(r2);
%   fprintf('NRMSE  pAD (regular): %.4f\n', e1);
%   fprintf('NRMSE  pAD_ss (ss)  : %.4f\n', e2);
% 
%   figure
%   subplot(131)
%   imagesc(fn_tile(d));  colorbar; clim(cl_res); axis equal; pbaspect([1,1,1]);
%   %title('target\_ss  |Mxy|');
%   title('target\_ss  |Mz|');
% 
%   subplot(132)
%   imagesc(fn_tile(r1)); colorbar; clim(cl_res); axis equal; pbaspect([1,1,1]);
%   title(sprintf('pAD (regular)\nNRMSE=%.3f', e1));
% 
%   subplot(133)
%   imagesc(fn_tile(r2)); colorbar; clim(cl_res); axis equal; pbaspect([1,1,1]);
%   title(sprintf('pAD\\_ss (ss-design)\nNRMSE=%.3f', e2));
% 
%   %sgtitle('Steady-state |Mxy|: OV90 vs OV90\_ss');
%   sgtitle('Steady-state |Mz|: OV90 vs OV90\_ss');
% end



