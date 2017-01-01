--[[
  Script for Training C3D in Torch
  Mohsen Fayyaz -- Sensifai -- Vision Group
--]]

-- Create table describing C3D Model configuration

require 'cudnn'
require 'nn'
require 'nnlr'

   local cfg = {64, 'M1', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
         elseif v == 'M1' then
            features:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
         else
            local oChannels = v;
            features:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1)
              :learningRate('weight',1)
              :learningRate('bias',2)
              :weightDecay('weight',1)
              :weightDecay('bias',0))
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   --features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*1*4*4))
   classifier:add(nn.Linear(512*1*4*4, 4096):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 101):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0)) --487 Sport1m-- UCF-101
   classifier:add(nn.LogSoftMax())

   c3dModel = nn.Sequential()
   c3dModel:add(features):add(classifier)
 

