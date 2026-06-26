function plotPulseComparison(pIni, pAD, dt_us,labels)
% PLOTPULSECOMPARISON  Compare initial and optimized mrphy.Pulse objects.
%
%   plotPulseComparison(pIni, pAD)
%   plotPulseComparison(pIni, pAD, dt_us)
%
%   Plots RF amplitude, RF phase, and Gx/Gy/Gz for pIni (blue) and pAD
%   (red) side-by-side in a 5-row figure.
%
%   Inputs:
%     pIni   - mrphy.Pulse, initial pulse  (rf: [1,nT] complex, gr: [3,nT])
%     pAD    - mrphy.Pulse, optimized pulse (same size as pIni)
%     dt_us  - (optional) dwell time in microseconds for the time axis.
%               If omitted, the x-axis is sample index.
%     labels - (optional) array of label strings for the two pulses,
%               If omitted, labels = ['Initial','Optimized']

nT = size(pIni.rf, 2);

if nargin < 4 || isempty(labels)
    labels = ["Initial", "Optimized"];
end

if nargin < 3 || isempty(dt_us)
    t = 1:nT;
    xlabel_str = 'Sample index';
else
    t = (0:nT-1) * dt_us * 1e-3;
    xlabel_str = 'Time (ms)';
end

rf_ini = pIni.rf(1,:);
rf_ad  = pAD.rf(1,:);
gr_ini = pIni.gr;   % [3, nT]
gr_ad  = pAD.gr;

amp_ini   = abs(rf_ini);
amp_ad    = abs(rf_ad);
phase_ini = angle(rf_ini) * 180/pi;
phase_ad  = angle(rf_ad)  * 180/pi;

chan_labels = {'Gx', 'Gy', 'Gz'};

figure('Name','Pulse Comparison','NumberTitle','off', ...
       'Units','normalized','Position',[0.05 0.05 0.6 0.88]);

%% --- RF Amplitude ---
subplot(5,1,1);
plot(t, amp_ini, 'b-',  'DisplayName',labels(1));  hold on;
plot(t, amp_ad,  'r--', 'DisplayName',labels(2));
ylabel('|RF| (a.u.)');
title('RF Amplitude');
legend('Location','best');
grid on; xlim([t(1) t(end)]);

%% --- RF Phase ---
subplot(5,1,2);
plot(t, phase_ini, 'b-',  'DisplayName',labels(1));  hold on;
plot(t, phase_ad,  'r--', 'DisplayName',labels(2));
ylabel('Phase (deg)');
title('RF Phase');
legend('Location','best');
grid on; xlim([t(1) t(end)]);

%% --- Gradients ---
for ch = 1:3
    subplot(5,1,2+ch);
    plot(t, gr_ini(ch,:), 'b-',  'DisplayName',labels(1));  hold on;
    plot(t, gr_ad(ch,:),  'r--', 'DisplayName',labels(2));
    ylabel('(a.u.)');
    title(chan_labels{ch});
    legend('Location','best');
    grid on; xlim([t(1) t(end)]);
end

xlabel(xlabel_str);

end
