function runDAQ(savedir, params)
%% main function for running the DAQ for fish and beads experiment
% The DAQ sends signals to 5 channels including Galvo, camera, DMD LED and RLED
% The output is synchronized 
% @author: @caichangjia

%% NiDAQ setup
dq = daq("ni");
dq.Rate = 8000;

%% signal generation
% params
channels = params.channels;   % representing Galvo, camera, DMD, LED, RLED sequentially, 1 is on, 0 is off
t = params.t;    % acquisition time sec
frOrig = params.frOrig; % origianl frame rate
cr = params.cr; % compression ratio        
galvoVolt = params.galvoVolt; % maximum voltage of galvo, reference 0.125
galvoMode = params.galvoMode; % linearOneWay for the streak camera, stepwise for shifted targeted frames, origin for still movies
offset = params.offset; % offset of galvo in seconds
LEDIntensity = params.LEDIntensity; % intensity of LED
LEDConstant = params.LEDConstant; % whether LED input in constant
outputM = [];

%% 
% galvo
if channels(1) == 1
    fprintf('Galvo is on \n');
    addoutput(dq, "Dev2", "ao0", "Voltage");  % add Galvo channel
    outGalvo = [];
    galvoOffset = round(-offset * dq.Rate); 
    if strcmp(galvoMode, 'linearOneWay')
        %outGalvo = [linspace(-galvoVolt, galvoVolt, dq.Rate/40)]; 
        outGalvo = [linspace(-galvoVolt, ((frOrig*t/cr)*2-1)*galvoVolt, t * dq.Rate)]; 
        for i = 1:t*dq.Rate
            if outGalvo(i) > galvoVolt + 1e-6
                outGalvo(i:end) = outGalvo(i:end) - 2 * galvoVolt;
                %disp(outGalvo(end))
            end        
        end
        outGalvo = outGalvo';
        %outGalvo = repmat(outGalvo, 1, t*40)'; 
        outGalvo = circshift(outGalvo, galvoOffset);
    elseif strcmp(galvoMode, 'stepwise')
        locs = linspace(-galvoVolt, galvoVolt, cr+1) + galvoVolt / cr;
        locs = locs(1:cr);
        locs = [locs, 0]; 
        outGalvo = repelem(locs, dq.Rate*0.05*1)';
    elseif strcmp(galvoMode, 'origin')
        outGalvo = repelem([0], dq.Rate * t)';
    end  
    outputM = [outputM, outGalvo];
end

% camera
if channels(2) == 1
    fprintf('Camera is on \n');
    addoutput(dq, "Dev2", "port0/line1", "Digital"); % Camera
    voltHigh = 1;
    dutyCycle = 0.1;
    if strcmp(galvoMode, 'stepwise')  % for estimating shifts
        rpc = dq.Rate * 0.2;
        high = zeros(dutyCycle * rpc, 1) + voltHigh;
        low = zeros((1-dutyCycle) * rpc, 1);
        outCamera = [high; low]; 
        cycles = t  / 0.2;
        outCamera = repmat(outCamera, ceil(round(cycles, 5)), 1);
        outCamera = outCamera(1:dq.Rate*t);
    else   % normal mode
        rpc = dq.Rate / frOrig * cr; 
        high = zeros(dutyCycle * rpc, 1) + voltHigh;
        low = zeros((1-dutyCycle) * rpc, 1);
        outCamera = [high; low]; 
        cycles = t * frOrig / cr;
        %outCamera = repmat(outCamera, cycles, 1);
        outCamera = repmat(outCamera, ceil(round(cycles, 5)), 1);
        outCamera = outCamera(1:dq.Rate*t);
    end
    outputM = [outputM, outCamera];
end

% DMD
if channels(3) == 1
    fprintf('DMD quick modulation is on \n');
    addoutput(dq, "Dev2", "port0/line3", "Digital"); % DMD
    voltHigh = 1;
    dutyCycle = 0.25;
    rpc = dq.Rate / frOrig / 5; 
    high = zeros(round(dutyCycle * rpc), 1) + voltHigh;
    low = zeros(round((1-dutyCycle) * rpc), 1);
    outDMD = [high; low]; 
    outDMD = repmat(outDMD, round(t * frOrig * 5), 1);
    outputM = [outputM, outDMD];
end

% LED
if channels(4) == 1
    fprintf('LED is on \n');
    addoutput(dq, "Dev2", "ao1", "Voltage");  % LED

    if LEDConstant == 1
        outLED = ones(t * dq.Rate, 1) * LEDIntensity';  
    else
        % frOrig = 50;
        % t = 5;
        % LEDIntensity = 0.3;
        rng(2023)
        voltBase = LEDIntensity; 
        voltBaseSig = LEDIntensity / 10;
        voltMax = LEDIntensity * 5;
        voltMaxSig = LEDIntensity * 5 / 10;
        tp = t * frOrig; 
        nSpikes = 10 * t; 
        tInterval = 0.01;        
        intensity = randn(tp, 1) * voltBaseSig + voltBase;
        spikes = randomNumbersWithMinDistance(tp, nSpikes, round(frOrig*tInterval));
        intensity(spikes) = intensity(spikes) + (randn(size(intensity(spikes))) * voltMaxSig + voltMax);
        x = 1:dq.Rate/frOrig:dq.Rate*t; 
        xq = 1:1:dq.Rate*t;
        intensity = interp1(x',intensity,xq', 'linear', voltBase);
        outLED = intensity;
    end
    outputM = [outputM, outLED];
end

% red LED
if channels(5) == 1
    addoutput(dq, "Dev2", "port0/line2", "Digital"); % Red LED
    fprintf('Red LED is on \n');
    voltHigh = 1;
    dutyCycle = 1; %0.005;  % 0.05 original
    frRedLED = 10; % RLED frequency per second
    rpc = dq.Rate / frRedLED; 
    timeIter = [10, 20]; % time for RLED light up and not light up for each iteration
    totalIter = ceil(t/sum(timeIter)); % total number of iteration during an experiment
    
    high = zeros(round(dutyCycle * rpc), 1) + voltHigh;
    low = zeros(round((1-dutyCycle) * rpc), 1);
    outRLED = [high; low];
    outRLED = repmat(outRLED, round(timeIter(1) * frRedLED), 1);
    outRLED = [zeros(timeIter(2)*dq.Rate, 1); outRLED]; % 10s off 10s on
    outRLED = repmat(outRLED, totalIter, 1);
    outRLED = outRLED(1:t * dq.Rate);
    outputM = [outputM, outRLED];
end
%save('C:\Users\nico\Desktop\data\zebrafish_6_22\wf_movie_2\outputM','outputM');

%% Plot the signal for debugging
% x = 0:1/dq.Rate:(t-1/dq.Rate);
% plot(x, outGalvo+0.5);
% hold on
% plot(x, outCamera-0.2, color='black');
% hold on 
% %plot(x, outDMD-0.8, '--', color='green');
% %hold on 
% %plot(x, outLED/max(outLED)*2, color='red');
% plot(x, outRLED/max(outRLED)*2, color='magenta');
% 
% %legend('Galvo signal', 'Camera signal', 'DMD signal', 'LED signal');
% legend('Galvo signal', 'Camera signal', 'RLED signal');
% xlim([0 0.1])
% xlabel('time(s)')
% ylabel('signal intensity')
% title('Output signal to DAQ')
% saveas(gcf, strcat(savedir, '\signal_to_daq.png'))

%% output the signal to the DAQ
write(dq, [outputM]); 

%% drive galvo mirror back to original position, turn off LED and rLED
clc
clear
dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev2", "ao0", "Voltage");  % Galvo
addoutput(dq, "Dev2", "ao1", "Voltage");  % LED
addoutput(dq, "Dev2", "port0/line2", "Digital");  % RLED
fprintf('Galvo returns to the original position \n');
fprintf('LED is off \n');
fprintf('RLED is off \n');
outGalvo1 = repelem([0], 1000)';
outLED = repelem([0], 1000)';
outRLED = repelem([0], 1000)';
write(dq, [outGalvo1, outLED, outRLED]);

%% save variables
%savedir = 'C:\Users\nico\Desktop\data\fluo_beads_11_2';
%save(strcat(savedir, '\beads_11_2.mat'))
end
