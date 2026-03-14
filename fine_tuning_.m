% fine-tune

clear
clc

load HI.mat
list = fieldnames(HI_extraction); %read all MCC fast-charging protocols
cells = fieldnames(HI_extraction.(list{ 1, 1})) ;
Input_base = mean(HI_extraction.(list{1, 1 }).(cells{ 1, 1 }).Results(:,[1,2,4]),2); % merge the most correlated three HIs as final HI
Output_base = HI_extraction.(list{1, 1}).(cells{1, 1}).Results(:, end); % read cell capacity

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

% save('tf_sourceMdl_opt.mat', 'mu_input', 'sigma_input', 'mu_output', 'sigma_output', ...
%     'Input_normalized', 'Output_normalized', 'layers', 'opts')
% 1. Error modeling
% 2. Transferred models

% clear

conditions = [3.7, 31, 5.9; 4.9 80, 4.9; 5, 67, 4; ...
    5.3, 54, 4; 5.6, 19, 4.6; 5.6, 36, 4.3; 5.9, 15, 4.6; ...
    5.9, 60, 3.1]; % typical parameters of MCC fast-charging protocols
load HI.mat
load Mdl_base.mat
load tf_sourceMdl_opt.mat
list = fieldnames(HI_extraction); %read all MCC fast-charging protocols

% for seedIdx = 1:10
for seedIdx = 1
    seedName = strcat('seed_', num2str(seedIdx));
    rng(seedIdx)

    time_transfer_base = tic;
    net = trainNetwork(Input_normalized', Output_normalized', layers, opts);
    time_transfer_base = toc(time_transfer_base);
    result = [Output_normalized, [predict(net, Input_normalized')]'];
    result = result*sigma_output + mu_output;
    TL_train.(list{1,1}).('cell7').data = result;

    netWhole = [net];
    list = fieldnames(Error);
    time_tf = [time_transfer_base];
    for i = 1:size(list, 1)
        cells = fieldnames(Error.(list{i,1}));
        Input_base = Error.(list{i, 1}).(cells{1, 1}).data(:, 1);
        Output_base = Error.(list{i, 1}).(cells{1, 1}).data(:, 2);
        Input_normalized = (Input_base-mu_input)/sigma_input;
        Output_normalized = (Output_base-mu_output)/sigma_output;
        if i == 1
            continue
        end
        layers = net.Layers;
        layers(end-1,1).WeightLearnRateFactor = 0;
        layers(end-1,1).BiasLearnRateFactor = 0;

        opts = trainingOptions("sgdm",...
    		"MaxEpochs", 500,...
    		"GradientThreshold", 1,...
    		"InitialLearnRate", 0.0025,...
    		"LearnRateSchedule","piecewise",...
    		"LearnRateDropPeriod", 125,...
    		"LearnRateDropFactor", 0.2, ...
    		"Verbose", 0);

        time_transfer = tic;
        netTransfer = trainNetwork(Input_normalized', Output_normalized', layers, opts);
        time_transfer = toc(time_transfer);

        netWhole = [netWhole, netTransfer];
        time_tf = [time_tf; time_transfer];
        result = [Output_normalized, [predict(netTransfer, Input_normalized')]'];
        result = result*sigma_output + mu_output;
        TL_train.(list{i,1}).(cells{1,1}).data = result;

    end
    % save(strcat('tf_mdls_',seedName,'.mat'), 'netWhole', "TL_train", "time_tf")
    disp([sum(time_tf), time_proposed_2 + time_proposed_1])
    
end

