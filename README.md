DBSCAN Clustering for Torch7
===================

A short binding to use DBSCAN in Torch7. wrapped from https://github.com/gyaikhom/dbscan

You have to have installed Torch7, then do this:

```bash
luarocks make
```

Example:

```lua
local dbscan = require 'dbscan'

local input = torch.DoubleTensor({{1,3,1},{1,4,1},{1,5,1},
                                  {1,6,1},{2,2,1},{2,3,0},
                                  {2,4,0},{2,5,0},{2,6,0}})

local epsilon, mininum = 1, 2

for i = 1, 2000 do
	sys.tic()
	local classid = dbscan.dbscan(input, epsilon, mininum)
	print(classid)
	print('dbscan: '..(sys.toc()*1000)..'ms')
end
```
