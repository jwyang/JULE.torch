----------------------------------------------------------------------------
-- This is the torch version for our CVPR 2016 paper: ----------------------
-- Joint Unsupervised Learning of Deep Representations and Image Clusters --
-- Authors: Jianwei Yang, Devi Parikh, Dhruv Batra. ------------------------
-- Contact jw2yang@vt.edu if you have any issues on running the code -------
----------------------------------------------------------------------------
require "hdf5"
require 'xlua'
require 'optim'
require 'cunn'
require 'cudnn'
require 'image'

local c                 = require 'trepl.colorize'
local affinity          = require 'affinity.affinity'
local evaluate          = require 'evaluate.evaluate'
local agg_clustering    = require 'agg_clustering.agg_clustering'
local criterion_triplet = require 'criterions.TripletEmbedding'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Joint Unsupervised Learning')
cmd:text()
cmd:text('Options')

-- two parameters that are most possible changed
cmd:option('-dataset','UMist','dataset name for evaluation')
cmd:option('-eta', 0.2, 'unrolling rate for recurrent process. 0.2 or 0.9, as described in the paper')

cmd:option('-epoch_rnn', 1, 'number of rnn epoches for joint learning')
cmd:option('-batchSize', 100, 'batch size for training CNN')

cmd:option('-learningRate', 0.01, 'base learning rate for training CNN')
cmd:option('-weightDecay', 5e-5, 'weight decay for training CNN')
cmd:option('-momentum', 0.9, 'momentum for training CNN')
cmd:option('-gamma_lr', 0.0001, 'gamma for inverse learning rate policy')
cmd:option('-power_lr', 0.75, 'power for inverse learning rate policy')

cmd:option('-num_nets', 1, 'number of models to train. Set it to be more than 1 get the statistics on performance, including mean and stddev.')
cmd:option('-epoch_pp', 20, 'number of CNN training epoch at each parially unrolled period. (20-50)')
cmd:option('-epoch_max', 1000, 'number of CNN training epoch in the whole recurrent process.')
cmd:option('-K_s', 20, 'number of neighbors for computing affinity between samples. (10-100)')
cmd:option('-K_c', 5, 'number of clusters for considering local structure. (1-10)')
cmd:option('-gamma_tr', 1, 'weight of positive pairs in weighted triplet loss. (1-2)')
cmd:option('-margin_tr', 0.2, 'margin for weighted triplet loss. (0.2-1.0)')
cmd:option('-num_nsampling', 20, 'number of negative samples for each positive pairs to construct triplet (1-20)')
cmd:option('-use_fast', 1, 'whether use fast affinity updating algorithm for acceleration. refer to the paper appendix section for detail (0 or 1)')
cmd:option('-updateCNN', 1, 'whether update CNN. set 0 to see the cllustering performance on raw image data and random CNN projections (0 or 1)')

cmd:option('-centralize_input', 0, 'centralize input image data')
cmd:option('-centralize_feature', 0, 'centralize output feature for clustering')
cmd:option('-normalize', 1, 'normalize output feature for clustering')

cmd:text()
local opt = cmd:parse(arg)
print(opt)

--------------------------
------ load dataset ------
--------------------------
print(c.blue '==>' ..' loading data')
local myFile = hdf5.open('datasets/'..opt.dataset..'/data4torch.h5', 'r')
local trainData_data  = myFile:read('data'):all():float()
local trainData_label = myFile:read('labels'):all():float()

-- centralize training data
if opt.centralize_input == 1 then
	local data_mean = torch.mean(trainData_data, 1)
	local xdata_mean = data_mean:new():expandAs(trainData_data)
	trainData_data:add(-1, xdata_mean)
end

local testData_data   = torch.FloatTensor(trainData_data:size()):copy(trainData_data)
local testData_label  = torch.DoubleTensor(trainData_label:size()):copy(trainData_label)

-----------------------------------
---- init networks parameters -----
-----------------------------------
local function NetInit(net)
	local function init(name)
		for k,v in pairs(net:findModules(name)) do
			v.weight:normal(0, 0.01)
			v.bias:zero()
		end
	end
	-- have to do for both backends
	init'cunn.SpatialConvolution'	
	init'cudnn.SpatialConvolution'	
	init'nn.SpatialConvolution'	
	init'nn.Linear'	
end

-----------------------------------------
----- convert label to label_table ------
-----------------------------------------
function cvt2TabelLabels(labels)
	-- derive the number of unique labels
	-- print({labels})
	labels_sorted, idx_sorted = torch.sort(labels)
    local nclasses = 1
    local label = labels_sorted[1]
    local labels_from_one = torch.LongTensor(labels:size(1)):zero()
    -- print("idx_sorted: ", idx_sorted[1])
    labels_from_one[idx_sorted[1]] = nclasses
    for i = 2, labels_sorted:size(1) do    
	    if labels_sorted[i] ~= label then
		    label = labels_sorted[i]
	        nclasses = nclasses + 1
        end
        labels_from_one[idx_sorted[i]] = nclasses
	end
    -- print('nclasses: ', nclasses)
    local labels_tb = {}
    for i = 1, nclasses do
        table.insert(labels_tb, {})
    end
    -- print(features:size(1))
    for i = 1, labels:size(1) do
        -- table.insert(labels_tb[labels[i]], i)
        table.insert(labels_tb[labels_from_one[i]], i)
    end
    return labels_tb
end


------------------------------------------------
----- initialize CNN models and variables ------
------------------------------------------------
print(c.blue '==>' ..' configuring model')
local num_networks           = opt.num_nets
local network_table          = {}
local parameters_table       = {}
local gradParameters_table   = {}
local optim_state_table      = {}
local label_gt_table_table   = {}
local label_gt_tensor_table  = {}
local label_pre_table_table  = {}
local label_pre_tensor_table = {}
print(num_networks)
local target_nclusters_table = torch.LongTensor(num_networks):zero()
local epoch_reset_labels     = torch.LongTensor(num_networks):zero()
for i = 1, num_networks do
	local model = nn.Sequential()
	model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
	model:add(dofile('models_def/'..opt.dataset..'.lua')):cuda()
	model:get(1).updateGradInput = function(input) return end
	table.insert(network_table, model)
    NetInit(network_table[i])
    local parameter, gradParameter = network_table[i]:getParameters()
    table.insert(parameters_table, parameter)
    table.insert(gradParameters_table, gradParameter)
    table.insert(optim_state_table, {}) 
	print(network_table[i])
	table.insert(label_pre_table_table, {})	
	table.insert(label_pre_tensor_table, {})
	table.insert(label_gt_table_table, {})
	table.insert(label_gt_tensor_table, {})	
end

for i = 1, num_networks do
	label_gt_tensor_table[i] = testData_label
	label_gt_table_table[i]  = cvt2TabelLabels(testData_label)
	target_nclusters_table[i] = #(label_gt_table_table[i])
end

-------------------------
----- set criterion -----
-------------------------
print(c.blue'==>' ..' setting criterion')
local criterion_triplet = nn.TripletEmbeddingCriterion(opt.margin_tr, opt.gamma_tr):cuda()

-------------------------
--- config optimizer ----
-------------------------
print(c.blue'==>' ..' configuring optimizer')
local optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
}
-----------------------------------------
------- get number of clusters ----------
-----------------------------------------
function getnClusters(label_pre)
	-- print(labels_pre)
	local nClusters = 0
	for i = 1, #label_pre do
		-- print(i, #(labels_pre[i]))
		if #(label_pre[i]) > 0 then
			nClusters = nClusters + 1
		end
	end
	return nClusters
end

---------------------------------------------
------- update image cluster labels ---------
---------------------------------------------
function updateLabels(features, label_pre, target_clusters, iter)
	print("compute affinity, ", features:size())
	local d, ind, W = affinity.compute(features, opt.K_s) --, W, L, sigma_l 
	-- sigma = sigma_l
	if iter == 0 then
	    print("initialize clusters...")
	    label_pre = agg_clustering.init(ind)
	    return label_pre
	end
    print("nclusters: ", getnClusters(label_pre))
	local A_us, A_s, label_pre = affinity.compute4cluster(features, W, label_pre, getnClusters(label_pre), target_clusters)
	print("nclusters: ", getnClusters(label_pre))
	local nClusters = getnClusters(label_pre)
	print("run agglomerative clustering...")
	local unfold_iter = torch.ceil((nClusters) * opt.eta)
	local unfold_valid_iter = (nClusters - target_clusters)
	local iterations = 0
	if unfold_iter < unfold_valid_iter then
		iterations = unfold_iter
	else
	    iterations = unfold_valid_iter
	end
	if iterations <= 0 then
	  	return label_pre
	end  
	label_pre = agg_clustering.run(W, A_us, A_s, label_pre, iterations, opt.K_c, opt.use_fast)
	return label_pre  
end

----------------------------------------------------
------- extract features for images from CNN -------
----------------------------------------------------
function extFeature(id_net)
	-- extract features from initial neural network
    network_table[id_net]:forward(trainData_data:index(1, torch.LongTensor{1, 2}))
    local dim_feature = network_table[id_net]:get(2):get(network_table[id_net]:get(2):size(1)).output:size(2)
    local features = torch.CudaTensor(trainData_data:size(1), dim_feature):zero()
  	local indices = torch.range(1, trainData_data:size(1)):long():split(opt.batchSize)   
	for t,v in ipairs(indices) do  
	    local inputs = trainData_data:index(1, v)
	    local outputs = network_table[id_net]:forward(inputs)
	    features:indexCopy(1, v, network_table[id_net]:get(2):get(network_table[id_net]:get(2):size(1)).output)
	end
	features = features:float()
	return features
end

---------------------------------------------
--- convert table labels to tensor labels ---
---------------------------------------------
function cvt2TensorLabels(label, ind_s, ind_e)
	local label_te = torch.FloatTensor(ind_e - ind_s + 1, 1):zero()
	for i = 1, #label do
		for j = 1, #(label[i]) do
			label_te[label[i][j]][1] = i
		end
	end
	return label_te
end

----------------------------------------
--- merging labels during training -----
----------------------------------------
function merge_label()	
	for i = 1, #network_table do
		local feature
		if epoch_reset_labels[i] == 0 or opt.updateCNN == 0 then
		    features = torch.Tensor(trainData_data:size()):copy(trainData_data):float()
		    features:resize(trainData_data:size(1), trainData_data:size(2) * trainData_data:size(3) * trainData_data:size(4))		 
		else
	        features = extFeature(i)
		end

		-- centralize
		if opt.centralize_feature == 1 then
			local feat_mean = torch.mean(features, 1)
			local xfeat_mean = feat_mean:new():view(1,features:size(2)):expand(features:size(1), features:size(2))
			features:add(-1, xfeat_mean)
		end

		-- normalize
		if opt.normalize == 1 then
			features:renorm(features, 2, 1, 1)
		end

	    print("feature dims: ", features:size())
	    label_pre_table_table[i] = updateLabels(features, label_pre_table_table[i], target_nclusters_table[i], epoch_reset_labels[i])	  
	    epoch_reset_labels[i] = epoch_reset_labels[i] + 1  
	    nclusters = #label_pre_table_table[i]
	    print("nclusters: ", nclusters)	    
    	label_pre_tensor_table[i] = cvt2TensorLabels(label_pre_table_table[i], 1, trainData_data:size(1))    	
	end	
end

------------------------------------------
----- Merging labels at final stage ------
------------------------------------------
function merge_label_final()
	local feature
	for i = 1, #network_table do
	    features = extFeature(i)	

		local myFile = hdf5.open('results/feature_pre_'..tostring(epoch)..'_'..tostring(i)..'.h5', 'w')
		myFile:write('feature', features:float())
		myFile:close()	

		-- centralize		
		if opt.centralize_feature == 1 then
			local feat_mean = torch.mean(features, 1)
			local xfeat_mean = feat_mean:new():view(1,features:size(2)):expand(features:size(1), features:size(2))
			features:add(-1, xfeat_mean)
		end

		-- normalize
		if opt.normalize == 1 then
		    features:renorm(features, 2, 1, 1)
		end

	    label_pre_table_table[i] = updateLabels(features, label_pre_table_table[i], target_nclusters_table[i], epoch_reset_labels[i])	  
	    epoch_reset_labels[i] = epoch_reset_labels[i] + 1  
	    nclusters = #label_pre_table_table[i]
	    print("nclusters: ", nclusters)	    
    	label_pre_tensor_table[i] = cvt2TensorLabels(label_pre_table_table[i], 1, trainData_data:size(1))    	
	end
end

----------------------------------
------- organize samples ---------
----------------------------------
function organize_samples(X, y)
	-- X: input features
	-- y: labels for input features
	local num_s = X:size(1)
    local y_table = cvt2TabelLabels(y)   
    -- print(y_table)
    local nclusters = #y_table
    if nclusters == 1 then
    	return
    else
    -- compute the size of triplet samples
    local num_neg_sampling = opt.num_nsampling
    if nclusters <= opt.num_nsampling then
	    local num_neg_sampling = nclusters - 1
	end

    local num_triplet = 0
    for i = 1, nclusters do
    	if #(y_table[i]) > 1 then
    		num_triplet = num_triplet + (#(y_table[i]) * (#(y_table[i]) - 1)) * num_neg_sampling / 2
    	end
    end
    if num_triplet == 0 then
		return
    end
    -- print('num_triplet: ', num_triplet)
    local A = torch.CudaTensor(num_triplet, X:size(2)):zero()
    local B = torch.CudaTensor(num_triplet, X:size(2)):zero()
    local C = torch.CudaTensor(num_triplet, X:size(2)):zero()
    local A_ind = torch.LongTensor(num_triplet):zero()
    local B_ind = torch.LongTensor(num_triplet):zero()
    local C_ind = torch.LongTensor(num_triplet):zero()    
    local id_triplet = 1
    for i = 1, nclusters do
    	if #(y_table[i]) > 1 then
	    	for m = 1, #(y_table[i]) do
	    		for n = m + 1, #(y_table[i]) do
				    if m ~= n then
    				    local is_choosed = torch.ShortTensor(num_s):zero()
    				    while 1 do
    				    	local rdn = torch.rand(1)
    				    	local id_s = torch.ceil(rdn[1] * num_s)
    				    	if is_choosed[id_s] == 0 and y[id_s] ~= y[y_table[i][m]] then
		    				    A_ind[id_triplet] = y_table[i][m]
		    				    B_ind[id_triplet] = y_table[i][n]
    				    		C_ind[id_triplet] = id_s
    				    		is_choosed[id_s] = 1
    				    		id_triplet = id_triplet + 1
    				    	end
					    	if (id_triplet) % num_neg_sampling == 1 then
    				    		break
    				    	end  
    				    end
			    	end
			    end
			end
    	end
    end    

    --print("id_triplet:", id_triplet)
    A:indexCopy(1, torch.range(1, num_triplet):long(), X:index(1, A_ind))
    B:indexCopy(1, torch.range(1, num_triplet):long(), X:index(1, B_ind))
    C:indexCopy(1, torch.range(1, num_triplet):long(), X:index(1, C_ind))
    return {A, B, C}, {A_ind, B_ind, C_ind}
  end
end

---------------------------------------
---- convert df_dtriplet to df_do -----
---------------------------------------
function cvt2df_do(df_do, df_dtriplets, triplets_ind)
	df_do:indexAdd(1, triplets_ind[1], df_dtriplets[1])
	df_do:indexAdd(1, triplets_ind[2], df_dtriplets[2])
	df_do:indexAdd(1, triplets_ind[3], df_dtriplets[3])
	return df_do
end

----------------------
----- udpate CNN -----
----------------------
function updateCNN()
	for i = 1, #network_table do
		network_table[i]:training()
	end
	epoch = epoch or 1	
	-- drop learning rate every "epoch_step" epochs
	print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']'..' [learningRate = ' .. optimState.learningRate .. ']')
	local targets = torch.CudaTensor(opt.batchSize)
	local indices = torch.randperm(trainData_data:size(1)):long():split(opt.batchSize)
    for t,v in ipairs(indices) do
    	local iter = epoch * (#indices) + t - 1
    	optimState.learningRate = opt.learningRate * torch.pow(1 + opt.gamma_lr * iter, - opt.power_lr)
    	targets = torch.CudaTensor(v:size(1))	    
	    -- xlua.progress(t, #indices)
	    local inputs = trainData_data:index(1,v)
	    for i = 1, #network_table do		 
		    targets:copy(label_pre_tensor_table[i]:index(1,v))   
	    	local feval = function(x)
	    	    if x ~= parameters_table[i] then parameters_table[i]:copy(x) end
	    	    gradParameters_table[i]:zero()	    	    
	    	    local outputs = network_table[i]:forward(inputs)
	    	    local triplets, triplets_ind = organize_samples(outputs, targets:float())	    	    
	    	    local f = 0
				if triplets ~= nil then
				    f = criterion_triplet:forward(triplets)
				    local df_dtriplets = criterion_triplet:backward(triplets)       
					local df_do = torch.CudaTensor():rand(outputs:size()):zero()
					df_do = cvt2df_do(df_do, df_dtriplets, triplets_ind)
					network_table[i]:backward(inputs, df_do)
				end		    	
		    	if t % 10 == 0 then
			        print("loss: ", f)
		        end
		    	return f,gradParameters_table[i]
		    end
		    optim.sgd(feval, parameters_table[i], optimState, optim_state_table[i])      
		end
	end
	epoch = epoch + 1
end

------------------------------------
------- evaluate performance -------
------------------------------------
function evalPerf()
	local nnsum = nn.Sum(1)
	local nnsm = nn.SoftMax()
	for i = 1, #network_table do
		network_table[i]:evaluate()
	end	
	print(c.blue '==>'.." testing")
	-- local bs = 100
	for i = 1, #network_table do		
		local myFile = hdf5.open('results/label_pre_'..tostring(epoch)..'_'..tostring(i)..'.h5', 'w')
		myFile:write('label', label_pre_tensor_table[i]:long())
		myFile:close()		
	    print('NMI: ' , evaluate.NMI(label_gt_table_table[i], label_pre_table_table[i], label_pre_tensor_table[i]:size(1)))
	end
end

------------------------------------
---- assert whether finished -------
------------------------------------
function is_allfinished()
	local flag = true
	for i = 1, #network_table do
		if #label_pre_table_table[i] > target_nclusters_table[i] then
			flag = false
		end
	end
	return flag
end

epoch = 0
optimState.learningRate = opt.learningRate
-- train multi-attribute discovery models
for n = 1, opt.epoch_rnn do
	for i = 0, opt.epoch_max do
		if i % opt.epoch_pp == 0 then			
			merge_label()
			evalPerf()    -- test mad models: show the clusters discovered by different model
			if is_allfinished() then
				break
			end
		end 
		if opt.updateCNN == 1 then
			updateCNN()       -- train mad models: train models with information-maximization objective while information minimization across models
		end
	end

	epoch_reset_labels:zero()
    while 1 do		
		merge_label_final()
		evalPerf()    -- test mad models: show the clusters discovered by different model
		if is_allfinished() then
			break
		end
	end
end
