local ffi = require 'ffi'

ffi.cdef[[
int dbscan_torch(THDoubleTensor* input, THIntTensor* classid, double epsilon, unsigned int minpts);
]]

local C = ffi.load(package.searchpath('libdbscan', package.cpath))

function dbscan.dbscan(input, epsilon, minpts)
	assert(torch.type(input) == 'torch.DoubleTensor')
	assert(torch.type(epsilon) == 'number')
	assert(torch.type(minpts) == 'number')
	local class = torch.IntTensor(input:size(1))
	C.dbscan_torch(input:cdata(), class:cdata(), epsilon, minpts)
	return class
end
