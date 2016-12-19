--[[
  Script for Training C3D in Torch
  Mohsen Fayyaz -- Sensifai -- Vision Group
--]]

--Preparing UCF101 For C3D Training

unpack = unpack or table.unpack

require 'nn'
require 'cunn'
require 'paths'
require 'torch'
require 'cutorch'
--require 'image'
local cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
require 'sys'
require 'lfs'
local posix = require 'posix'



local function main()
 
  

  dataPathPre = '/home/snf/mohsen/UCF101/UCF-101/'
  spilitPathPre = '/home/snf/mohsen/UCF101/ucfTrainTestlist/'
  ucfSpilit = 'trainlist02.txt'
  
    
  --fmeans = torch.load('fmeans.dat')
 
  inputData = {}
    

  i = 0
  for line in io.lines(spilitPathPre..ucfSpilit) do
    i = i+1
    j = 0
    inputData[i] = {}
    tmp = {}
    for splt in string.gmatch(line,"%S+") do
      j = j+1
      table.insert(tmp, splt)
      if(j==1) then
        inputData[i][j] = tmp[1]

      else
        inputData[i][j] = tonumber(tmp[2])
      --	print(tmp[2])
      end
    end
    --print(i)
  end
  trainSamples = i


  trainList = {}
    
  numBatch = 16
  cropSize = 112
  newHeight = 128
  newWidth = 171
    
    
  print(trainSamples)

  trainList = {}
  L = 0
  
  dir = 'clips/'.. ucfSpilit:sub(1,-4) ..'/'
  if not (posix.stat(dir, "type") == 'directory') then
    lfs.mkdir(dir)
  end
    
    
for ln = 1,trainSamples do
      --ln = math.fmod(t-1,trainSamples) + 1
	lineNumber = ln
	print('Line: ', lineNumber, '/ ', trainSamples)
	sampleLine = inputData[lineNumber][1]
	print(sampleLine)
	
	label = inputData[lineNumber][2]
	
	
	continue = false
	if posix.stat(dataPathPre..inputData[lineNumber][1])==nil then
   		print("Failed to open video. Skiped it")
		--os.exit(0)
		continue = true
	end
  
  --clip = torch.CudaHalfTensor(augSize, 3, numBatch, cropSize, cropSize)
	
	if(not continue) then
    cap  = cv.VideoCapture{dataPathPre..inputData[lineNumber][1]}
    nF = cap:get{cv.CAP_PROP_FRAME_COUNT}
    FOURCC = cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}--cap:get{cv.CAP_PROP_FOURCC}
    FPS = cap:get{cv.CAP_PROP_FPS}
    
		nFrame = 0
    nClp = 0
    nF = nF - math.fmod(nF,numBatch)
		
    
    f = 0
    
    print(nF)      
    for b=0,nF-1,numBatch do
			
      outFileC = dir..ln..'_'..label..'_'..(b)..'_C.avi'
          
          writer = cv.VideoWriter{outFileC, FOURCC, FPS,frameSize={cropSize,cropSize}}
          
        outFileF = dir..ln..'_'..label..'_'..(b)..'_F.avi'
          
          writerF = cv.VideoWriter{outFileF, FOURCC, FPS,frameSize={cropSize,cropSize}}
      
      
      for f=b,b+numBatch-1 do 
        --print(f)
        cap:set{1,f}
        ret,frame = cap:read{}
        --print(ret)
        --HWC2CHW
        --frame = frame:permute(3,1,2)
        
        --AugData
        if(not ret) then
          writer:release()
          writerF:release()
          os.remove(outFileC)
          os.remove(outFileF)
          print('Frame Corrupted! Skipping the clip')
          break
        end
        --print(f..'/'..nF)
        frame = cv.resize{frame,{newWidth,newHeight}}
        frame = cv.getRectSubPix{frame, patchSize={cropSize,cropSize}, center={newWidth/2, newHeight/2}}
        flip = cv.flip{src=frame, flipCode = 1}
            
        writer:write{frame}
        writerF:write{flip}
          
      end  
      writer:release()
      writerF:release()
      
      trainList[#trainList+1] = {}
      trainList[#trainList][1] = outFileC
      trainList[#trainList][2] = label
      
      trainList[#trainList+1] = {}
      trainList[#trainList][1] = outFileF
      trainList[#trainList][2] = label
      --collectgarbage()
      --print('clip')
		end
      
	end
	
end

torch.save(ucfSpilit:sub(1,-4)..".t7",trainList)

end
main()

