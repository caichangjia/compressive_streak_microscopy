%% script for the beads experiment
% @author: @caichangjia
% channels % representing Galvo, camera, DMD, LED, RLED sequentially, 1 is on, 0 is off
% t: acquisition time sec
% frOrig: origianl frame rate, 100/200/400 Hz for streak movie
% cr: compression ratio, 10        
% galvoVolt: maximum voltage of galvo, 0.04
% galvoMode: linearOneWay for the streak camera, stepwise for shifted targeted frames, origin for still movies
% offset: offset of galvo in seconds, 1.25e-4
% LEDIntensity: intensity of LED
% LEDConstant: whether LED input in constant
%% adjust LED power for imaging static beads
dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev2", "ao1", "Voltage");  % LED
outLED = ones(dq.Rate*1, 1) * 1;
write(dq, [outLED]); 

%% infer shifts with different galvo voltage
savedir = NaN;
for cr = [5, 10, 15, 20, 25]
    for galvoVolt = [0.02, 0.04, 0.06, 0.08, 0.10]
        cr
        galvoVolt
        params = struct('t', (cr+1)*0.025*1, 'frOrig', 400, 'cr', cr, 'galvoVolt', galvoVolt, ...
            'galvoMode', 'stepwise', 'offset', 5e-4, 'LEDIntensity', 1.25); % each frame exposure time 0.025 s
        runDAQ(savedir, params)
    end
end

%% test
params = struct('t', 3, 'frOrig', 400, 'cr', 10, 'galvoVolt', 0.08, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 0.5, 'LEDConstant', 1);
runDAQ(savedir, params)

%% performance vs compression ratio
cr = [10, 15, 20, 25];
cr = cr(4)
galvoVolt = 0.08; 
params = struct('t', 3, 'frOrig', 400, 'cr', cr, 'galvoVolt', galvoVolt, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 1.00, 'LEDConstant', 1);
runDAQ(savedir, params)


%% performance vs galvo input volt
cr = 10;
for galvoVolt = [0.02, 0.04, 0.06, 0.08, 0.10]  
    params = struct('t', 3, 'frOrig', 400, 'cr', cr, 'galvoVolt', galvoVolt, ...
        'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 1.00, 'LEDConstant', 1);
    runDAQ(savedir, params)
end

%% performance vs LED intensity
for intensity = [0.25, 0.5, 0.75, 1, 1.25]    
    params = struct('t', 3, 'frOrig', 400, 'cr', 10, 'galvoVolt', 0.08, ...
        'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', intensity, 'LEDConstant', 1);
    runDAQ(savedir, params)
end