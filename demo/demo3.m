function demo3()
  disp('============================ demo3.m ================================');
  disp('This tutorial demos an design of a b0/b1 robust inversion pulse using');
  disp('the AutoDiffPulses tool with an adiabatic full passage RF as init');
  disp('The design assums a max b1 of 0.25 Gauss.');
  disp('It accounts for b0 of range [-200, 200] Hz, and 10-percent b1 loss.');
  disp('============================ demo3.m ================================');

  [flim, b1lim] = deal([-200, 200], [0.9, 1]);  % Hz; a.u.
  [nf, nb] = deal(201, 10);
  b1max = 0.25;  % Gauss

  adiabatic_opt(flim, b1lim, nf, nb, b1max);

end

function adiabatic_opt(flim, b1lim, nf, nb, b1max)
  [target, cube, pIni, b1Map] = init(flim, b1lim, nf, nb, b1max);

  err_meth = 'l2z';

  pIni0 = copy(pIni);
  for ii = 1:1
    [pIni1, optInfo] = adpulses.opt.arctanAD(...
                        target, cube, pIni0, 'err_meth', err_meth ...
                        , 'doClean',false, 'gpuID',1, 'fName','adpulses_inv' ...
                        , 'niter_gr',0, 'niter',10, 'pen_meth','null' ...
                        , 'b1Map',b1Map);
    rf = pIni1.rf;
    rf = (rf + rf(end:-1:1))/2;
    pIni1.rf = rf;
    pIni0 = copy(pIni1);
  end
  pAD = copy(pIni1);

  pulse_c = {pIni, pAD};
  title_c = {'pIni', 'pAD'};

  plot_res(pulse_c, title_c, cube, target, b1Map);
end

function [target, cube, pulse, b1Map] = init(flim, b1lim, nf, nb, b1max)
  import attr.*
  arg = mrphy.utils.envMR('get_s');
  [dt0, gam, rfmax] = getattrs(arg, {'dt','gam','rfmax'});

  %% 
  [fov, ofst] = deal([0,0,0], [0,0,0]); % cm
  imSize = [1, nb, nf];
  tmp1 = ones(imSize);
  b0Map = bsxfun(@times, tmp1, reshape(linspace(flim(1),flim(2),nf),1,1,[]));
  b1Map = bsxfun(@times, tmp1, linspace(b1lim(1), b1lim(2), nb));
  weight = ones(imSize);

  fn_target = @(d, weight)struct('d',d, 'weight',weight);

  d = cat(4, zeros(imSize), zeros(imSize), -ones(imSize));

  target = fn_target(d, weight);

  cube = mrphy.SpinCube(fov, imSize, ofst, 'b0Map',b0Map);

  %% Pulse
  dt = 4e-6;
  beta = 5.3;  % a.u., dflt for typical adiabatic pulse

  rf_peak = 0.8*b1max;

  fn_adiabatic = @fullpassage;

  tp = 1.5e-3; % Sec
  bw = 0.8e3;  % Hz

  rf = rf_peak * fn_adiabatic(tp, beta, bw, dt);  % Gauss
  gr = zeros(3, numel(rf));
  pulse = mrphy.Pulse('rf',rf, 'gr',gr, 'rfmax',b1max, 'dt',dt);
end

function rf_n = fullpassage(tp, beta, bw, dt)
  tn = linspace(-1, 1, 2*round(tp/dt/2)+1);  % ensure #tn is odd
  cSub = mrphy.utils.ctrSub(numel(tn));
  amp = sech(beta * tn);
  amp = amp / max(amp);

  frq = bw/2 * tanh(beta*-tn);

  phs = cumsum(frq * dt * pi);
  phs = phs - phs(cSub) + 0*pi/2;

  rf_n = amp .* exp(1i * phs);
end

%% Utils
function plot_res(pulse_c, title_c, cube, target, b1Map)
  fn_sim = @(p)cube.applypulse(p, 'doCim',true, 'doEmbed',true, 'b1Map',b1Map);
  b0_ = cube.b0Map;
  % spread spins w/ same b0 different b1 to different b0
  tmp = reshape((0:size(b0_,2)-1)/size(b0_,2)*abs(b0_(2)-b0_(1)), 1, []);
  b0Map = bsxfun(@plus, b0_, tmp);

  nP = numel(pulse_c);
  MT_c = cell(1, nP);
  fn_t = @(p)(1:numel(p.rf))*p.dt;

  % fn_res = @(MT)MT(:,:,:,1) + 1i*MT(:,:,:,2);
  fn_res = @(MT)MT;
  fn_prep = @(P)reshape(P, [], 3); % -> (nb, xyz)
  % fn_prep = @(P)P;  % place holder

  fn_rf = @(rf)[abs(rf(:)), real(rf(:)), imag(rf(:))]; 

  d = fn_res(target.d);
  d(cube.mask == 0) = nan;
  nan_mask = isnan(d);

  d(nan_mask) = 0;
  d = fn_prep(d);

  fn_cmplx = @(x)x(:,1) + 1i*x(:,2);

  d_cmplx = fn_cmplx(d);

  fn_err = @(x)sqrt(nansum(abs((d(:,3)-x(:,3)).*target.weight(:)).^2));

  for ii = 1:nP
    pulse = copy(pulse_c{ii});
    rf = 1.0 * pulse.rf;

    pulse.rf = 1.0 * rf;
    MT_c{ii} = fn_res(fn_sim(pulse));
    MT_c{ii}(nan_mask) = 0;
    MT_c{ii} = fn_prep(MT_c{ii});
  end

  figure
  subplot(211)
  plot(b0Map(:), d); grid on; xlim([min(b0Map(:)), max(b0Map(:))])
  title('target magnetizations');
  legend({'Mx', 'My', 'Mz'});
  ylim([-1.1, 1.1])

  for ii = 1:nP
    figure
    subplot(211)
    plot(b0Map(:), MT_c{ii}); grid on; xlim([min(b0Map(:)), max(b0Map(:))])
    title([title_c{ii}, ' magnetizations: ', num2str(fn_err(MT_c{ii}))]);
    legend({'Mx', 'My', 'Mz'});
    ylim([-1.1, 1.1])
    xlabel('freq (Hz)');

    subplot(212)
    plot(fn_t(pulse_c{ii}), fn_rf(1.0*pulse_c{ii}.rf)); grid on
    xlabel('time (Sec)');
    ylabel('rf (Gauss)');
    legend({'abs', 'real', 'imag'});
  end
end

