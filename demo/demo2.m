function demo()

IniVar = matfile('./IniVars.mat');

cube = IniVar.cube;

OV90(cube, IniVar.target_OV90, IniVar.pIni_OV90);

end

function OV90(cube, target, pIni)
  pIni_20 = pIni.interpT(20e-6); % new dt
  names_c = {'adpulses_opt_arctanAD', 'adpulses_opt_arctanAD_20'};

  fncs_c = {@adpulses.opt.arctanAD, @adpulses.opt.arctanAD};
  args_c = {{target, cube, pIni, 'err_meth', 'ml2xy', 'doClean',false ...
             , 'gpuID',1, 'fName',names_c{1}},
            {target, cube, pIni_20, 'err_meth', 'ml2xy', 'doClean',false ...
             , 'gpuID',2, 'fName',names_c{2}}};
  slns_c = {cell(1,2), cell(1,2)};

  parfor ii = 1:2
    [slns_c{ii}{:}] = fncs_c{ii}(args_c{ii}{:});
  end

  [pAD, optInfo] = deal(slns_c{1}{:});
  [pAD_20, optInfo_20] = deal(slns_c{2}{:});

  pAD_20to4 = pAD_20.interpT(4e-6);

  pulse_c = {pAD, pAD_20, pAD_20to4};
  title_c = {'pAD', 'pAD_20', 'pAD_20to4'};

  figure
  plot_res(pulse_c, title_c, cube, target, 'xy');

  disp('OV90');

  keyboard
end

%% Utils
function plot_res(pulse_c, title_c, cube, target, xy_z)
  fn_sim = @(p)cube.applypulse(p, 'doCim',true, 'doEmbed',true);

  nP = numel(pulse_c);
  MT_c = cell(1, nP);

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
  nan_mask = isnan(d);

  d(nan_mask) = 0;
  d = fn_tile(d);

  for ii = 1:nP
    MT_c{ii} = fn_res(fn_sim(pulse_c{ii}));
    MT_c{ii}(nan_mask) = 0;
    MT_c{ii} = fn_tile(MT_c{ii});
  end

  figure
  imagesc(d); colorbar; 
  caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title('target');

  for ii = 1:nP
    figure
    imagesc(MT_c{ii}); colorbar;
    caxis(cl_res); axis('equal'); pbaspect([1,1,1]); title(title_c{ii});
  end

end

function P = tile3dto2d(P)
  [nx, ny, nz] = size(P);
  nz_rt = sqrt(nz);
  [nc, nr] = deal(ceil(nz_rt), floor(nz_rt));
  P = cat(3, P, zeros([nx, ny, nr*nc-nz])); % -> (nx, ny, nc*nr)
  P = reshape(P, nx, ny*nc, []); % -> (nx, nc*ny, nr)
  P = reshape(permute(P, [2, 1, 3]), ny*nc, nx*nr).'; % -> (nr*nx, nc*ny);
end

