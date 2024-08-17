%% this script test LED 
% @author: @caichangjia
dq = daq("ni");
dq.Rate = 8000;
addoutput(dq, "Dev2", "ao1", "Voltage");  % LED
addoutput(dq, "Dev2", "port0/line2", "Digital"); % Red LED

%% signal generation
% params
t = 60;    % acquisition time sec
constantLED = 1; % whether to supply LED with constant voltage
voltHigh = 1;
dutyCycle = 1;  % 0.05 original
frRedLED = 10; % RLED frequency per second
rpc = dq.Rate / frRedLED; 
timeIter = [10, 10]; % time for RLED light up and not light up for each iteration
totalIter = ceil(t/sum(timeIter)); % total number of iteration during an experiment

high = zeros(round(dutyCycle * rpc), 1) + voltHigh;
low = zeros(round((1-dutyCycle) * rpc), 1);
outRLED = [high; low];
outRLED = repmat(outRLED, round(timeIter(1) * frRedLED), 1);
outRLED = [zeros(timeIter(2)*dq.Rate, 1); outRLED];
outRLED = repmat(outRLED, totalIter, 1);
outRLED = outRLED(1:t * dq.Rate);
plot(outRLED)
outLED = [zeros(size(outRLED))];  

%% output the signal to the DAQ
write(dq, [outLED, outRLED]); 
