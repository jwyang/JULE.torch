require 'nn'
require 'cunn'

local backend_name = 'nn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local net = nn.Sequential()

-- building block
local function ConvBNReLU(module, nInputPlane, nOutputPlane)
  module:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 5, 5, 1, 1, 0, 0))
  module:add(nn.SpatialBatchNormalization(nOutputPlane))
  module:add(backend.ReLU(true))
  return module
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = backend.SpatialMaxPooling

-- containing multiple sequentials
local nseqs = 2
local nInputPlanes = {3, 50}
local nOutputPlanes = {50, 50}
for i = 1, nseqs do
  module = nn.Sequential()
  ConvBNReLU(module, nInputPlanes[i], nOutputPlanes[i])
  module:add(MaxPooling(2,2,2,2):ceil())
  net:add(module)
end
-- In the last block of convolutions the inputs are smaller than
-- the kernels and cudnn doesn't handle that, have to use cunn
backend = nn
net:add(nn.View(9800))
net:add(nn.Linear(9800,160))
net:add(nn.Normalize(2))

return net
