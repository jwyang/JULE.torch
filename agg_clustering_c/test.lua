#!/usr/bin/lua
require("agg_clustering")
A = torch.FloatTensor():rand(5,3)
compute_CAff(A)
