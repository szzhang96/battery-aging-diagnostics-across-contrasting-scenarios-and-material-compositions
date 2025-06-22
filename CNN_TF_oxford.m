% Transfer learning

clearvars -except seedIdx percentage Estimation_results cellNameG seedName
rng(seedIdx)

load HI.mat
load(strcat('D:\STUDY\06.SOH_base_error\BRANDNEW\MIT\tf_mdls_',seedName,'.mat'))

i = 1;
cellNameG = fieldnames(HI_all_2);
whole_size = size(HI_all_2.(cellNameG{i,1}), 1);
num_extraction = ceil(whole_size*percentage);
random_indices = sort(randperm(whole_size, num_extraction));

HI_real = HI_all_2.(cellNameG{i,1});
Q_real = Q_all_2.(cellNameG{i,1});

Input_base = HI_real(random_indices,:); 
Output_base = Q_real(random_indices, :);

m1 = mean(Input_base);
s1 = std(Input_base);
m2 = mean(Output_base);
s2 = std(Output_base);

Input_normalized = (Input_base - m1)/s1;
Output_normalized = (Output_base - m2)/s2;

layers = netWhole(1).Layers;
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
    fprintf('%.4f\n', time_transfer)

result = [Output_normalized, [predict(netTransfer, Input_normalized')]'];
        result = result*s2 + m2;

        save('TL_mdl_train_oxford.mat', "netTransfer", "result")
    
%% Verification on test data: other cells of all MCC fast-charging protocols
MAPE = [];
RMSE = [];

for i = 1:8
	HI = HI_all_2.(cellNameG{i, 1});
	HI_normalized = (HI-m1)/s1;
	Q_real = Q_all_2.(cellNameG{i,1});
	Q_est = predict(netTransfer, HI_normalized')*s2+m2;
	RE = (Q_est' - Q_real)./Q_real*100;
    E = (Q_est' - Q_real);
	mape = mean(abs(E)./Q_real*100);
	rmse = rms(E);
	MAPE = [MAPE; mape];
	RMSE = [RMSE;rmse];
	Estimation_results.(seedName).transfer.(cellNameG{i, 1}).Results = [Q_real, Q_est', RE];
end
Estimation_results.(seedName).transfer.MAPE_final = MAPE;
Estimation_results.(seedName).transfer.RMSE_final = RMSE;
Estimation_results.(seedName).transfer.trainTime = time_transfer;
% disp(RMSE')
% 
save('EstimationResults_oxford.mat', 'Estimation_results')
