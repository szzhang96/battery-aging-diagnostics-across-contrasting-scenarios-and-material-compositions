% 1. Base model
% 2. Transfer learning basic model

clear
clc

load HI.mat
list = fieldnames(HI_extraction); %read all MCC fast-charging protocols
cells = fieldnames(HI_extraction.(list{ 1, 1})) ;
Input_base = mean(HI_extraction.(list{1, 1 }).(cells{ 1, 1 }).Results(:,[1,2,4]),2); % merge the most correlated three HIs as final HI
Output_base = HI_extraction.(list{1, 1}).(cells{1, 1}).Results(:, end); % read cell capacity
time_proposed_1 = tic;
MDl_base = polyfit(Input_base, Output_base, 1); % base modeling via linear curve titting
time_proposed_1 = toc(time_proposed_1);

baseTrainResult = polyval(MDl_base, Input_base);
save('Mdl_base.mat', "MDl_base", "time_proposed_1", "baseTrainResult", "Output_base", "Input_base")

mu_input = mean(Input_base);
sigma_input = std(Input_base);
Input_normalized = (Input_base-mu_input)/sigma_input;

mu_output = mean(Output_base);
sigma_output = std(Output_base);
Output_normalized = (Output_base-mu_output)/sigma_output;

layers = [
    sequenceInputLayer(size(Input_base, 2))
    convolution1dLayer(10,10,'Padding','causal')
    convolution1dLayer(30,30,'Padding','causal')
    fullyConnectedLayer(1)
    regressionLayer];

opts = trainingOptions("adam",...
    "MaxEpochs", 1000,...
    "GradientThreshold", 1,...
    "InitialLearnRate", 0.005,...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropPeriod", 125,...
    "LearnRateDropFactor", 0.2, ...
    "Verbose", 0);

save('tf_sourceMdl_opt.mat', 'mu_input', 'sigma_input', 'mu_output', 'sigma_output', ...
    'Input_normalized', 'Output_normalized', 'layers', 'opts')
