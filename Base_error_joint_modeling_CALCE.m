% Base-error modeling

clear
clc

for seedIdx = 1:10
	seedName = strcat('seed_', num2str(seedIdx));
	rng(seedIdx)

    load HI.mat
    load('D:\STUDY\06.SOH_base_error\BRANDNEW\MIT\Mdl_base.mat')

    i = 1;
    cellNameG = fieldnames(HI_all_2);
    whole_size = size(HI_all_2.(cellNameG{i,1}), 1);
	percentage = 0.05;
	num_extraction = ceil(whole_size*percentage);
	random_indices = sort(randperm(whole_size, num_extraction));

    HI_real = HI_all_2.(cellNameG{i,1});
	Q_real = Q_all_2.(cellNameG{i,1});
	
	selected_HI = HI_real(random_indices,:); 
	selected_Q = Q_real(random_indices, :);

    save('selectedData.mat', "Q_real", "HI_real", "selected_Q", "selected_HI")

	Estimation_base = polyval(MDl_base, selected_HI);
	error = selected_Q - Estimation_base;
	
	len = size(error, 1);
    Input_error = Estimation_base;
    Output_error = error;
	
	time_proposed = tic;
	MDl_error = fitrgp(Input_error, Output_error, 'KernelFunction', 'ardmatern32');
	time_proposed = toc(time_proposed);
    fprintf('%.4f\n', time_proposed)
	errTrain_output = predict(MDl_error, Input_error);
	
    trainResult = [selected_Q, Estimation_base, errTrain_output, Estimation_base+errTrain_output];
    save('trainResult.mat', "trainResult")

	%% Verification on test data: other cells of all MCC fast-charging protocols
	MAPE_base = [];
	RMSE_base = [];
	MAPE_final = [];
	RMSE_final = [];

	for i = 1:4
		HI = HI_all_2.(cellNameG{i, 1});
		Q_real = Q_all_2.(cellNameG{i,1});
		Q_base = polyval(MDl_base, HI);
        E_base = Q_base - Q_real;
		RE_base = (Q_base - Q_real)./Q_real * 100;
		mape_base = mean(abs(E_base)./Q_real*100);
		rmse_base = sqrt(mean(E_base.^2));
		MAPE_base = [MAPE_base; mape_base];
		RMSE_base = [RMSE_base; rmse_base];

		Q_error = predict(MDl_error, Q_base);

		Q_est = Q_base + Q_error;
		RE_final = (Q_est - Q_real)./Q_real*100;
        E_final = Q_est - Q_real;
		mape_final = mean(abs(E_final)./Q_real*100);
		rmse_final = rms(E_final);
		MAPE_final = [MAPE_final; mape_final];
		RMSE_final = [RMSE_final;rmse_final];
		Estimation_results.(seedName).proposed.(cellNameG{i, 1}).Results = ...
            [Q_real, Q_base, RE_base, Q_error, Q_est, abs(E_final)];
	end
	Estimation_results.(seedName).proposed.MAPE_base = MAPE_base;
	Estimation_results.(seedName).proposed.RMSE_base = RMSE_base;
	Estimation_results.(seedName).proposed.MAPE_final = MAPE_final;
	Estimation_results.(seedName).proposed.RMSE_final = RMSE_final;
	Estimation_results.(seedName).proposed.trainTime = time_proposed;
	% disp(RMSE_final')
	run CNN_TF_CALCE.m
end
run resultProcess_CALCE.m
