
function genericSetup(doSavePath) %#ok<FNDEF> % generic, named intentionally
% This function always assume itself locates in the root dir of the package
if ~exist('doSavePath','var'), doSavePath = false; end
theDir = fileparts(mfilename('fullpath'));
prevDir = cd(theDir);

%% MATLAB dependencies (auto-bootstrap)
% Clone the steady-state-enabled MATLAB packages into ./matlab_deps at pinned
% versions, then compile the C Bloch simulator.
% This folder is git-ignored; delete it to force a fresh checkout.
depsDir = fullfile(theDir, 'matlab_deps');
if ~exist(depsDir, 'dir'), mkdir(depsDir); end

setupDep(fullfile(depsDir, '+mrphy'), ...
         'https://github.com/tianrluo/MRphy.mat.git', ...
         'v0.2.1');
setupDep(fullfile(depsDir, '+attr'), ...
         'https://github.com/fmrilab/attr.mat.git', '');

compileBlochcim(fullfile(depsDir, '+mrphy', '+sims'));

addpath(depsDir);  % parent dir of the +mrphy / +attr packages

%% generate path
% don't use pwd in genpath, as names in dName_rm_c can appear in it.
dName_s = genpath('.'); % _s: string, not string type though
% appending path w/ filesep makes fn_mtchd no need to worry partial matching
dName_s = strrep(dName_s, pathsep, [filesep,pathsep]);
dName_c = strsplit(dName_s, pathsep)'; % _c: cell
dName_c(end) = []; % dName_s ends w/ pathsep, trailing dName_c an empty {}, rm.

% customize dName_rm_c for different usages
% 'matlab_deps' holds package folders (+mrphy, +attr); only their parent dir
% belongs on the path, which is added explicitly above.
dName_rm_c = {'demo', 'private', '.git', 'test', 'tests', 'arch', 'back', ...
              'adpulses', 'matlab_deps'};

fn_mtchd = @(x,y)~isempty(strfind(x,[y,filesep])); %#ok<STREMP>
for iName = 1:numel(dName_rm_c)
  mask = cellfun(@(x)fn_mtchd(x, dName_rm_c{iName}), dName_c);
  dName_c(mask) = [];
end

%% add path
warning('off', 'MATLAB:mpath:packageDirectoriesNotAllowedOnPath');
% 1st char, '.', and last char, filesep, removed before addpath
for iName = 1:numel(dName_c), addpath([theDir, dName_c{iName}(2:end-1)]); end
warning('on', 'MATLAB:mpath:packageDirectoriesNotAllowedOnPath');

%% save?
if doSavePath, savepath; end

disp('Make sure you have also setup the python codes, checkout `setup.py`.')
%% head back
cd(prevDir);
end

% =========================================================================
function setupDep(targetDir, url, commit)
% Clone `url` into `targetDir` (named so MATLAB sees it as the right package)
% and, if given, checkout the pinned `commit`. No-op if already present.
  if exist(targetDir, 'dir')
    fprintf('[setup] dependency present, skipping clone: %s\n', targetDir);
    return;
  end
  fprintf('[setup] cloning %s -> %s\n', url, targetDir);
  if system(sprintf('git clone "%s" "%s"', url, targetDir)) ~= 0
    error(['git clone failed for %s.\n', ...
           'Ensure `git` is installed and the URL is reachable, or clone ', ...
           'it manually to %s.'], url, targetDir);
  end
  if ~isempty(commit)
    if system(sprintf('git -C "%s" checkout --quiet %s', targetDir, commit)) ~= 0
      warning('Could not checkout pinned commit %s in %s.', commit, targetDir);
    end
  end
end

% =========================================================================
function compileBlochcim(simsDir)
% Compile the C Bloch simulator (blochcim.c) with mex, unless already built.
  src = fullfile(simsDir, 'blochcim.c');
  if ~exist(src, 'file')
    warning('blochcim.c not found at %s; skipping mex compile.', src);
    return;
  end
  if ~isempty(dir(fullfile(simsDir, ['blochcim.', mexext])))
    fprintf('[setup] blochcim already compiled, skipping.\n');
    return;
  end
  fprintf('[setup] compiling %s\n', src);
  prev = cd(simsDir);
  try
    mex blochcim.c
  catch ME
    msg = sprintf(['mex compile failed: %s\n', ...
                   'Compile manually with `mex blochcim.c` in %s.'], ...
                  ME.message, simsDir);
    warning('%s', msg);
  end
  cd(prev);
end
