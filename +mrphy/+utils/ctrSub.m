function cSub = ctrSub(Nd, varargin)
% As a separate fn, ensure consistent behaviour of getting subscripts to the
% center of a Nd Matrix
% 0 1 2 3 4 5 6
% 0 1 2 2 3 3 4
% for consistency w/ fftshift and ifftshift, where the location cSub[1] shifted
% to is assumed as the center

if nargin == 0, test(); return; end

Nd = [Nd, varargin{:}];

cSub = ceil((Nd+1)/2);
cSub(Nd == 0) = 0;

end

%%
function test()
prefix = mfilename('fullpath');
disp('------------------------');
disp([prefix, '.test()']);

assert(all([0,1,2,2,3,3,4] == mrphy.utils.ctrSub(0:6)));

disp([prefix, '.test() passed']);
end
