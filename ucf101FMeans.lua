require 'thffmpeg'
require 'sys'
require 'math'
require "try-catch"

x = THFFmpeg()
y = THFFmpeg()
x = nil
collectgarbage()
collectgarbage()

dataPathPre = '/home/deepface/Fayyaz/UCF101/UCF-101/'
spilitPathPre = '/home/deepface/Fayyaz/UCF101/ucfTrainTestlist/'
ucfSpilit = 'trainlist01.txt'

data = {}

i = 0
for line in io.lines(spilitPathPre..ucfSpilit) do
	i = i+1
	j = 0
	data[i] = {}
	tmp = {}
	for splt in string.gmatch(line,"%S+") do
		j = j+1
		table.insert(tmp, splt)
		if(j==1) then
			data[i][j] = tmp[1]

		else
			data[i][j] = tonumber(tmp[2])
		--	print(tmp[2])
		end
	end
end


local fmeans = {0,0,0}
n=0
for k=1,i do
	print(k,'/',i)
	continue = false
	if not y:open(dataPathPre..data[k][1]) then
   		print("Failed to open video. Skiped it")
		--os.exit(0)
		continue = true
	end
	if(not continue) then
		while true do
			frame = y:next_frame()
			if(frame == nil) then
				break
			end
			fmeans[1] = fmeans[1] + frame[1]:mean()
			fmeans[2] = fmeans[2] + frame[2]:mean()
			fmeans[3] = fmeans[3] + frame[3]:mean()
			n = n+1
		end
	end

end

--os.execute('sleep 10')
fmeans[1] = fmeans[1] / n
fmeans[2] = fmeans[2] / n
fmeans[3] = fmeans[3] / n

torch.save('fmeans.dat',fmeans)
print("Means ok")
