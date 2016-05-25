--------------------------------------------------------------------------------
-- TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
-- Jianwei Yang, Dec 16
--------------------------------------------------------------------------------

local TripletEmbeddingCriterion, parent = torch.class('nn.TripletEmbeddingCriterion', 'nn.Criterion')

function TripletEmbeddingCriterion:__init(margin, gamma)
   parent.__init(self)
   self.margin = margin or 0.5
   self.gamma = gamma or 2
   self.Li = torch.Tensor()
   self.d_pos = torch.Tensor()
   self.d_neg = torch.Tensor()
   self.gradInput = {}
end

function TripletEmbeddingCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.Li:resize(N)
   self.d_pos:resize(N)
   self.d_neg:resize(N)

   local delta_pos = a - p
   local delta_net = a - n

   local norm_delta_pos = torch.norm(delta_pos, 2, 2)
   local norm_delta_neg = torch.norm(delta_net, 2, 2)

   norm_delta_pos:cmul(norm_delta_pos):mul(self.gamma)
   norm_delta_neg:cmul(norm_delta_neg)

   local delta_pos_neg = (norm_delta_pos - norm_delta_neg):add(self.margin)

   for i = 1, N do
      self.Li[i] = math.max(0, delta_pos_neg[i][1])
   end
   self.output = self.Li:sum() / N
   return self.output
end

function TripletEmbeddingCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   if torch.type(a) == 'torch.CudaTensor' then -- if buggy CUDA API
      self.gradInput[1] = (a:mul(self.gamma - 1) + n - p:mul(self.gamma)):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[2] = (p - a):mul(self.gamma):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
   else -- otherwise
      self.gradInput[1] = self.Li:gt(0):diag():type(a:type()) * (a:mul(self.gamma - 1) + n - p:mul(self.gamma)) * 2/N
      self.gradInput[2] = self.Li:gt(0):diag():type(a:type()) * (p - a):mul(self.gamma) * 2/N
      self.gradInput[3] = self.Li:gt(0):diag():type(a:type()) * (a - n) * 2/N
   end
   return self.gradInput
end
