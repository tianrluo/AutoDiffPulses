function demo()

IniVar = matfile('./IniVars.mat');

cube = IniVar.cube;

OV90(cube, IniVar.target_OV90, IniVar.pIni_OV90);

IV180(cube, IniVar.target_IV180, IniVar.pIni_IV180);

IV180M(cube, IniVar.target_IV180M, IniVar.pIni_IV180M);

end

function OV90(cube, target, pIni)
  pAD = adpulses.opt.arctanAD(target, cube, pIni, 'err_meth', 'ml2xy' ...
                              , 'doClean',false, 'gpuID',0);

  figure
  plot_res(pIni, pAD, cube, target, 'xy');
  sgtitle('OV90');
end

function IV180(cube, target, pIni)
  pAD = adpulses.opt.arctanAD(target, cube, pIni, 'err_meth', 'l2z' ...
                              , 'doClean',false, 'gpuID',0);
   
  figure
  plot_res(pIni, pAD, cube, target, 'z');
  sgtitle('IV180');
end

function IV180M(cube, target, pIni)
  pAD = adpulses.opt.arctanAD(target, cube, pIni, 'err_meth', 'l2z' ...
                              , 'doClean',false, 'gpuID',0);
 
  figure
  plot_res(pIni, pAD, cube, target, 'z');
  sgtitle('IV180M');
end

%% Utils
function plot_res(p1, p2, cube, target, xy_z)
  fn_sim = @(p)cube.applypulse(p, 'doCim',true, 'doEmbed',true);
  [MT_1, MT_2] = deal(fn_sim(p1), fn_sim(p2));
  
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
  d(cube.mask == 0) = nan;
  [MT_1, MT_2] = deal(fn_res(MT_1), fn_res(MT_2));
  d(isnan(d)) = 0;
  MT_1(isnan(MT_1)) = 0;
  MT_2(isnan(MT_2)) = 0;

  % reshapes
  [d, MT_1, MT_2] = deal(fn_tile(d), fn_tile(MT_1), fn_tile(MT_2));
  
  subplot(131);
  imagesc(d); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title('target');
  
  subplot(132);
  imagesc(MT_1); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title('Initial');
  
  subplot(133);
  imagesc(MT_2); colorbar;
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title('Proposed');
end

function P = tile3dto2d(P)
  [nx, ny, nz] = size(P);
  nz_rt = sqrt(nz);
  [nc, nr] = deal(ceil(nz_rt), floor(nz_rt));
  P = cat(3, P, zeros([nx, ny, nr*nc-nz])); % -> (nx, ny, nc*nr)
  P = reshape(P, nx, ny*nc, []); % -> (nx, nc*ny, nr)
  P = reshape(permute(P, [2, 1, 3]), ny*nc, nx*nr).'; % -> (nr*nx, nc*ny);
end
