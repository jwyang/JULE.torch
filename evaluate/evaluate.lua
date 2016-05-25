---
--- FUnctions to evaluate the performance of clustering algorithm.
---

local evaluate = {}

-- > labels_gt, table of ground-truth labels
-- > labels_pre, table of predicted labels
-- > N, total number of samples
-- < NMI
function evaluate.NMI(labels_gt, labels_pre, N)
   -- compute entropy for labels_gt
   local pr_gt = torch.FloatTensor(#labels_gt, 1):zero()   
   for i = 1, #labels_gt do
      pr_gt[i] = #(labels_gt[i]) / N
   end
   local pr_gt_log = torch.log(pr_gt)
   local H_gt = -torch.sum(torch.cmul(pr_gt, pr_gt_log))
   
   -- compute entropy for labels_pre
   -- print("size:", #labels_pre)
   local pr_pre = torch.FloatTensor(#labels_pre, 1):zero()   
   for i = 1, #labels_pre do
      pr_pre[i] = #(labels_pre[i]) / N
   end
   local pr_pre_log = torch.log(pr_pre)
   local H_pre = -torch.sum(torch.cmul(pr_pre, pr_pre_log))

   -- compute mutual information
   -- build M_gt
   local M_gt = torch.FloatTensor(N, #labels_gt):zero()
   for i = 1, #labels_gt do
      for j = 1, #(labels_gt[i]) do
         -- print(labels_gt[i][j])
         M_gt[labels_gt[i][j]][i] = 1
      end      
   end

   -- build M_pre
   local M_pre = torch.FloatTensor(N, #labels_pre):zero()
   for i = 1, #labels_pre do
      for j = 1, #(labels_pre[i]) do
         M_pre[labels_pre[i][j]][i] = 1
      end      
   end
   local pr_gp = torch.mm(M_gt:t(), M_pre) / N
   pr_gp_log = torch.log(pr_gp + 1e-10)
   H_gp = -torch.sum(torch.cmul(pr_gp, pr_gp_log))

   -- compute mutual information
   local MI = H_gt + H_pre - H_gp

   local NMI = MI / torch.sqrt(H_gt * H_pre)

   return NMI
end

return evaluate
 
