--
-- The agglomerative clustering algorithm.
--
require 'hdf5'

local agg_clustering = {}
local K_c = 5
-- > indices: indice for k-nearest neighbors
-- < labels: cluster labels for samples
function agg_clustering.init(indices)
   -- initialize labels for input data given knn indices
   local nsamples = (#indices)[1]
   local k = (#indices)[2]
   local visited = torch.IntTensor(nsamples, 1):fill(-1)
   local count = 0
   for i = 1, nsamples do
      local cur_idx = i
      local pos = {}
      while visited[cur_idx][1] == -1 do
         table.insert(pos, cur_idx)
         local neighbor = 0
         for k = 1, (indices[cur_idx]:size(1)) do
            neighbor = indices[cur_idx][k]
            -- print(cur_idx, neighbor)
            -- print(k)
            if cur_idx ~= neighbor then
               break;
            end
         end
         visited[cur_idx][1] = -2
         cur_idx = neighbor         
         if #pos > 50 then  
            break;
         end
      end
      if visited[cur_idx][1] < 0 then
         visited[cur_idx][1] = count
         count = count + 1
      end
      for j = 1, #pos do
         visited[pos[j]][1] = visited[cur_idx][1]
      end
   end
   -- print(count)
   local label_indice = {}
   for i = 1, count do
   	table.insert(label_indice, {})
   end
   for i = 1, nsamples do
      table.insert(label_indice[visited[i][1] + 1], i) -- (label_indice[visited[i][1]] or 0) + 1
   end
   for i = 1, count do
      if #(label_indice[i]) == 0 then
         print("error")
      end
   end
   -- error()
   -- print(label_indice)
   return label_indice
end

function agg_clustering.merge_two_clusters(W, A_s_t, A_us_t, Y_t, idx_c_a, idx_c_b)
   nclusters = #Y_t

   A_us_t:indexAdd(2, torch.LongTensor{idx_c_a}, A_us_t:index(2, torch.LongTensor{idx_c_b}))
   
   -- update A_t(i->idx_c_a) = r_a * A_t(i->idx_c_a) + r_b * A_t(i->idx_c_b) (fast algorithm)
   -- nsamples in cluster idx_c_a
   
   local nsamples_c_a = (#Y_t[idx_c_a])
   local nsamples_c_b = (#Y_t[idx_c_b])
   local ratio = nsamples_c_a / (nsamples_c_a + nsamples_c_b)
   A_us_t:select(1, idx_c_a):mul(ratio)
   A_us_t:select(1, idx_c_b):mul(1 - ratio)
   A_us_t:indexAdd(1, torch.LongTensor{idx_c_a}, A_us_t:index(1, torch.LongTensor{idx_c_b}))

   A_us_t[idx_c_a][idx_c_a] = 0
   A_us_t:select(2, idx_c_b):zero()
   A_us_t:select(1, idx_c_b):zero()

   -- update A_t(i->idx_c_a)
   -- update cluster labels Y_t   
   for k = 1, #(Y_t[idx_c_b]) do
      Y_t[idx_c_a][#(Y_t[idx_c_a]) + 1] = Y_t[idx_c_b][k]
   end
   Y_t[idx_c_b] = {}

   -- update A_s_t   
   for i = 1, nclusters do
      if #(Y_t[i]) == 0 or i == idx_c_a then
         A_s_t[i][idx_c_a] = 0
         A_s_t[idx_c_a][i] = 0
      elseif i < idx_c_a then
         A_s_t[i][idx_c_a] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      elseif i > idx_c_a then
         A_s_t[idx_c_a][i] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      end
   end
   return A_s_t, A_us_t, Y_t
end

function agg_clustering.search_clusters(A_s_t)
   -- print("cluster numbers:", nclusters)
   local A_sorted, Idx_sort = torch.sort(A_s_t, 1, true)
   local aff = torch.FloatTensor(1, A_sorted:size(2)):zero()
   for i = 1, A_sorted:size(2) do
      aff[1][i] = A_sorted[1][i]
      if A_sorted:size(2) > 100 then
         for k = 2, K_c do
            aff[1][i] = aff[1][i] + (A_sorted[1][i] - A_sorted[k][i]) / (K_c - 1)
         end
      end
   end

   local v_c, idx_c = torch.max(aff, 2) -- each row
   -- find corresponding cluster labels for two clusters
   local idx_c_b = idx_c[1][1]         -- col
   local idx_c_a = Idx_sort[1][idx_c_b]        -- row

   if idx_c_a == idx_c_b then
      print("error")
      error()
   elseif idx_c_a > idx_c_b then
      local temp = idx_c_a
      idx_c_a = idx_c_b
      idx_c_b = temp
   end
   return idx_c_a, idx_c_b
end

function agg_clustering.run_step(W, A_s_t, A_us_t, Y_t)
   -- timer = torch.Timer()
   -- get the number of clusters
   nclusters = #Y_t
   -- print("Cluster Num: ", nclusters)
   -- find maximal value in A_t

   local idx_c_a, idx_c_b = agg_clustering.search_clusters(A_s_t)
   -- print('merge pairs: ', idx_c_a, idx_c_b)
   -- update affinity matrix A_t
   -- update A_t(idx_c_a->i) = A_t(idx_c_a->i) + A_t(idx_c_b->i)
   A_us_t:indexAdd(2, torch.LongTensor{idx_c_a}, A_us_t:index(2, torch.LongTensor{idx_c_b}))

   -- update cluster labels Y_t   
   for k = 1, #Y_t[idx_c_b] do
      Y_t[idx_c_a][#(Y_t[idx_c_a]) + 1] = Y_t[idx_c_b][k]
   end
   Y_t[idx_c_b] = {}
   
   -- update A_t(i->idx_c_a)
   
   for i = 1, nclusters do
      if #(Y_t[i]) > 0 and i ~= idx_c_a then
         local W_i = W:index(1, torch.LongTensor(Y_t[i]))
         local W_i_idx_c_a = W_i:index(2, torch.LongTensor(Y_t[idx_c_a]))
         local W_idx_c_a = W:index(1, torch.LongTensor(Y_t[idx_c_a]))
         local W_idx_c_a_i = W_idx_c_a:index(2, torch.LongTensor(Y_t[i]))
         -- print(W_idx_c_a_i:size())
         -- print(W_i_idx_c_a:size())
         -- print(#(Y_t[idx_c_a]))
         A_us_t[idx_c_a][i] = torch.sum(torch.mm(W_idx_c_a_i, W_i_idx_c_a))      
      end
   end
   
   -- print(A_us_t)
   A_us_t[idx_c_a][idx_c_a] = 0
   A_us_t:select(2, idx_c_b):zero()
   A_us_t:select(1, idx_c_b):zero()

   local nclusters = #Y_t   
   for i = 1, nclusters do
      if #(Y_t[i]) == 0 or i == idx_c_a then
         A_s_t[i][idx_c_a] = 0
         A_s_t[idx_c_a][i] = 0
      elseif i < idx_c_a then
         A_s_t[i][idx_c_a] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      elseif i > idx_c_a then
         A_s_t[idx_c_a][i] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      end
   end
   A_s_t:select(1, idx_c_b):zero()
   A_s_t:select(2, idx_c_b):zero()
   -- return updated A_s_t, A_us_t and Y_t
   return A_s_t, A_us_t, Y_t
end

function agg_clustering.run_step_fast(W, A_s_t, A_us_t, Y_t)
   -- timer = torch.Timer()
   -- get the number of clusters
   local nclusters = #Y_t
   -- print("Cluster Num: ", nclusters)
   -- find maximal value in A_t
   local idx_c_a, idx_c_b = agg_clustering.search_clusters(A_s_t)
   -- update affinity matrix A_t
   -- update A_t(idx_c_a->i) = A_t(idx_c_a->i) + A_t(idx_c_b->i)
   A_us_t:indexAdd(2, torch.LongTensor{idx_c_a}, A_us_t:index(2, torch.LongTensor{idx_c_b}))
   
   -- update A_t(i->idx_c_a) = r_a * A_t(i->idx_c_a) + r_b * A_t(i->idx_c_b) (fast algorithm)
   -- nsamples in cluster idx_c_a
   
   A_us_t:indexAdd(1, torch.LongTensor{idx_c_a}, A_us_t:index(1, torch.LongTensor{idx_c_b}))   
   -- update A_t(i->idx_c_a)
   -- update cluster labels Y_t   
   for k = 1, #(Y_t[idx_c_b]) do
      Y_t[idx_c_a][#(Y_t[idx_c_a]) + 1] = Y_t[idx_c_b][k]
   end
   Y_t[idx_c_b] = {}
   
   -- update A_s_t   
   for i = 1, nclusters do
      if #(Y_t[i]) == 0 or i == idx_c_a then
         A_s_t[i][idx_c_a] = 0
         A_s_t[idx_c_a][i] = 0
      elseif i < idx_c_a then
         A_s_t[i][idx_c_a] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      elseif i > idx_c_a then
         A_s_t[idx_c_a][i] =  A_us_t[idx_c_a][i] / torch.pow(#(Y_t[idx_c_a]), 2) + A_us_t[i][idx_c_a] / torch.pow(#(Y_t[i]), 2)
      end
   end

   -- print(A_us_t:size())
   -- print(nclusters)
   if idx_c_b ~= nclusters then
      -- print(idx_c_b)
      -- print(A_us_t:index(1, torch.LongTensor{1}))
      A_us_t:indexCopy(1, torch.LongTensor{idx_c_b}, A_us_t:index(1, torch.LongTensor{nclusters}))
      A_us_t:indexCopy(2, torch.LongTensor{idx_c_b}, A_us_t:index(2, torch.LongTensor{nclusters}))
      A_us_t[idx_c_b][idx_c_b] = 0

      -- print("Pre: ", A_s_t:sub(1, idx_c_b, idx_c_b, idx_c_b))
      -- print("Pre: ", A_s_t:sub(idx_c_b, idx_c_b, idx_c_b, nclusters))
      A_s_t:sub(1, idx_c_b, idx_c_b, idx_c_b):copy(A_s_t:sub(1, idx_c_b, nclusters, nclusters))      
      A_s_t:sub(idx_c_b, idx_c_b, idx_c_b, nclusters):copy(A_s_t:sub(idx_c_b, nclusters, nclusters, nclusters):t())
      A_s_t[idx_c_b][idx_c_b] = 0
      -- print("Cur: ", A_s_t:sub(1, idx_c_b, idx_c_b, idx_c_b))
      -- print("Cur: ", A_s_t:sub(idx_c_b, idx_c_b, idx_c_b, nclusters))

      for k = 1, #(Y_t[nclusters]) do
         Y_t[idx_c_b][#(Y_t[idx_c_b]) + 1] = Y_t[nclusters][k]
      end
   end

   A_us_t = A_us_t:sub(1, nclusters - 1, 1, nclusters - 1)   
   A_s_t = A_s_t:sub(1, nclusters - 1, 1, nclusters - 1)
   table.remove(Y_t, nclusters)   
   -- print(Y_t)
   -- timer = torch.Timer()
   -- print('Time-2 elapsed: ' .. timer:time().real .. ' seconds')
   -- return updated A_s_t, A_us_t and Y_t
   return A_s_t, A_us_t, Y_t
end

--  > W: MxM affinity matrix, where M is the number of samples
--  > Y_0: {N} table, whose elements is the positions for one cluster
--  > verbose: prints a progress bar or not
--
--  < Y_T, predicted labels for X after T timesteps
function agg_clustering.run(W, A_unsym_0, A_sym_0, Y_0, T, K_c_in, use_fast)
   -- compute initial affinity among clusters\
   local nclusters = #Y_0
   A_sym_0_sum = torch.sum(A_sym_0, 1)

   K_c = K_c_in   
   -- update affinity among clusters and Y as well
   local t = 0
   timer = torch.Timer()   
   while t < T do
      if use_fast == 1 then
         A_sym_0, A_unsym_0, Y_0 = agg_clustering.run_step_fast(W, A_sym_0, A_unsym_0, Y_0)
      else
         A_sym_0, A_unsym_0, Y_0 = agg_clustering.run_step(W, A_sym_0, A_unsym_0, Y_0)
      end
      t = t + 1
   end
   print('Time elapsed for agg clustering: ' .. timer:time().real .. ' seconds')
   local Y_T = {}
   for i = 1, #Y_0 do
      if #(Y_0[i]) > 0 then
         table.insert(Y_T, Y_0[i])
      end
   end
   return Y_T
end

return agg_clustering
