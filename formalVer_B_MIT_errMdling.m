% 1. Error modeling
% 2. Transferred models

clear

conditions = [3.7, 31, 5.9; 4.9 80, 4.9; 5, 67, 4; ...
    5.3, 54, 4; 5.6, 19, 4.6; 5.6, 36, 4.3; 5.9, 15, 4.6; ...
    5.9, 60, 3.1]; % typical parameters of MCC fast-charging protocols
load HI.mat
load Mdl_base.mat
load tf_sourceMdl_opt.mat
list = fieldnames(HI_extraction); %read all MCC fast-charging protocols

for seedIdx = 1:10
    seedName = strcat('seed_', num2str(seedIdx));
    rng(seedIdx)

    Input_error = []; %input data of Error model: base capacity and typical parameters of MCC fast-charging protocols
    Output_error = []; % output data of Error model: estimation error of base capacity
    for i = 1:size(list, 1)
        cells = fieldnames(HI_extraction.(list{i,1}));
        whole_size = size(HI_extraction.(list{i,1}).(cells{1,1}).Results, 1);
        percentage = 0.01;
        num_extraction = ceil(whole_size*percentage);
        random_indices = sort(randperm(whole_size, num_extraction));
        selected_HI = mean(HI_extraction.(list{i,1}).(cells{1, 1}).Results(random_indices,[1,2,4]), 2);
        selected_Q = HI_extraction.(list{i,1}).(cells{1, 1}).Results(random_indices, end);
        Estimation_base = polyval(MDl_base, selected_HI);
        error = selected_Q - Estimation_base;

        len = size(error, 1);
        Input_error = [Input_error; [Estimation_base, conditions(i,:).*ones(len, 3)]];
        Output_error = [Output_error; error];
    end
    time_proposed_2 = tic;
    Mdl_error = fitrgp(Input_error, Output_error, 'KernelFunction', 'ardmatern32');
    time_proposed_2 = toc(time_proposed_2);

    for i = 1:size(list, 1)
        cells = fieldnames(HI_extraction.(list{i,1}));
        whole_size = size(HI_extraction.(list{i,1}).(cells{1,1}).Results, 1);
        percentage = 0.01;
        num_extraction = ceil(whole_size*percentage);
        random_indices = sort(randperm(whole_size, num_extraction));
        selected_HI = mean(HI_extraction.(list{i,1}).(cells{1, 1}).Results(random_indices,[1,2,4]), 2);
        selected_Q = HI_extraction.(list{i,1}).(cells{1, 1}).Results(random_indices, end);
        Estimation_base = polyval(MDl_base, selected_HI);
        error = selected_Q - Estimation_base;

        len = size(error, 1);
        Input_error = [Estimation_base, conditions(i,:).*ones(len, 3)];
        Output_error = error;

        errTrainResult = predict(Mdl_error, Input_error);

        Error.(list{i,1}).(cells{1,1}).data = [selected_HI, selected_Q];
        Error.(list{i,1}).(cells{1,1}).Error_model = [Estimation_base, error, errTrainResult];
    end

    save(strcat('Error_model_',seedName,'.mat'), "Error", 'Mdl_error',"time_proposed_2");

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
    save(strcat('tf_mdls_',seedName,'.mat'), 'netWhole', "TL_train", "time_tf")
    disp([sum(time_tf), time_proposed_2 + time_proposed_1])
end
