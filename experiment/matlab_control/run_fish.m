%% script for the fish experiment
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

%% test
savedir = NaN;
params = struct('channels', [1, 1, 0, 1, 0], 't', 60, 'frOrig', 5, 'cr', 1, 'galvoVolt', 0.08, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 0.5, 'LEDConstant', 1);
runDAQ(savedir, params)


%% widefield movie
savedir = NaN;
params = struct('channels', [0, 1, 0, 1, 0], 't', 60, 'frOrig', 5, 'cr', 1, 'galvoVolt', 0.08, ...    
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 0.3, 'LEDConstant', 1);
runDAQ(savedir, params)

%% targeted movie
savedir = NaN;
params = struct('channels', [0, 1, 0, 1, 0], 't', 60, 'frOrig', 5, 'cr', 1, 'galvoVolt', 0.08, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 8.0, 'LEDConstant', 1);
runDAQ(savedir, params)

%% streak movie 10Hz
savedir = NaN;
params = struct('channels', [1, 1, 0, 1, 0], 't', 60, 'frOrig', 100, 'cr', 10, 'galvoVolt', 0.04, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 8.0);
runDAQ(savedir, params)

%% streak movie 20Hz
savedir = NaN;
params = struct('channels', [1, 1, 0, 1, 0], 't', 60, 'frOrig', 200, 'cr', 10, 'galvoVolt', 0.04, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 8.0, 'LEDConstant', 1);
runDAQ(savedir, params)

%% streak movie 40Hz
savedir = NaN;
params = struct('channels', [1, 1, 0, 1, 0], 't', 60, 'frOrig', 400, 'cr', 10, 'galvoVolt', 0.04, ...
    'galvoMode', 'linearOneWay', 'offset', 1.25e-4, 'LEDIntensity', 8, 'LEDConstant', 1);
runDAQ(savedir, params)

%% infer shifts with different galvo voltage, usually used on targeted image
savedir = NaN;
for cr = [10, 15, 20]
    for galvoVolt = [0.04, 0.06, 0.08]
        cr
        galvoVolt
        params = struct('channels', [1, 1, 0, 1, 1], 't', (cr+1)*0.05*1, 'frOrig', 50, 'cr', cr, 'galvoVolt', galvoVolt, ...
            'galvoMode', 'stepwise', 'offset', 5e-4, 'LEDIntensity', 1.25, 'LEDConstant', 1);
        runDAQ(savedir, params)
    end
end
