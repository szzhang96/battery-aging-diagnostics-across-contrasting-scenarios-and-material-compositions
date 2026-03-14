% base-error

clear
clc

load .\data\MIT\HI.mat
HI_MIT = HI_extraction;
load .\data\oxford\HI.mat
HI_oxford = HI_all_2;
Q_oxford = Q_all_2;
load .\data\CALCE\HI.mat
HI_CALCE = HI_all_2;
Q_CALCE = Q_all_2;


list = fieldnames(HI_MIT); %read all MCC fast-charging protocols
cells = fieldnames(HI_MIT.(list{ 1, 1})) ;
Input_base = mean(HI_MIT.(list{1, 1 }).(cells{ 1, 1 }).Results(:,[1,2,4]),2); % merge the most correlated three HIs as final HI
Output_base = HI_MIT.(list{1, 1}).(cells{1, 1}).Results(:, end); % read cell capacity
mu_in_MIT = mean(Input_base);
std_in_MIT = std(Input_base);
mu_out_MIT = mean(Output_base);
std_out_MIT = std(Output_base);
Input_base   = (Input_base - mu_in_MIT) ./ std_in_MIT;
Output_base   = (Output_base - mu_out_MIT) ./ std_out_MIT;
time_proposed_1 = tic;
MDl_base = polyfit(Input_base, Output_base, 1); % base modeling via linear curve titting
time_proposed_1 = toc(time_proposed_1);

baseTrainResult = polyval(MDl_base, Input_base);
% save('Mdl_base.mat', "MDl_base", "time_proposed_1", "baseTrainResult", "Output_base", "Input_base")


conditions = [3.7, 31, 5.9; 4.9 80, 4.9; 5, 67, 4; ...
    5.3, 54, 4; 5.6, 19, 4.6; 5.6, 36, 4.3; 5.9, 15, 4.6; ...
    5.9, 60, 3.1]; % typical parameters of MCC fast-charging protocols

for seedIdx = 1:10
    seedName = strcat('seed_', num2str(seedIdx));
    rng(seedIdx)

    Input_error = []; %input data of Error model: base capacity and typical parameters of MCC fast-charging protocols
    Output_error = []; % output data of Error model: estimation error of base capacity

    % MIT
    for i = 1:size(list, 1)
        cells = fieldnames(HI_MIT.(list{i,1}));
        whole_size = size(HI_MIT.(list{i,1}).(cells{1,1}).Results, 1);
        percentage = 0.01;
        num_extraction = ceil(whole_size*percentage);
        random_indices = sort(randperm(whole_size, num_extraction));
        selected_HI = mean(HI_MIT.(list{i,1}).(cells{1, 1}).Results(random_indices,[1,2,4]), 2);
        selected_Q = HI_MIT.(list{i,1}).(cells{1, 1}).Results(random_indices, end);

        selected_HI   = (selected_HI - mu_in_MIT) ./ std_in_MIT;
        selected_Q   = (selected_Q - mu_out_MIT) ./ std_out_MIT;

        Estimation_base = polyval(MDl_base, selected_HI);
        error = selected_Q - Estimation_base;

        len = size(error, 1);
        Input_error = [Input_error; [Estimation_base, [conditions(i,:), 1].*ones(len, 4)]];
        Output_error = [Output_error; error];
    end
    
    % oxford
    i = 1;
    cellNameG_oxford = fieldnames(HI_oxford);
    whole_size = size(HI_oxford.(cellNameG_oxford{i,1}), 1);
	percentage = 0.1;
	num_extraction = ceil(whole_size*percentage);
	random_indices = sort(randperm(whole_size, num_extraction));

    HI_real = HI_oxford.(cellNameG_oxford{i,1})/1000;
	Q_real = Q_oxford.(cellNameG_oxford{i,1})/1000;
	
	selected_HI = HI_real(random_indices,:); 
	selected_Q = Q_real(random_indices, :);
    
    mu_in_oxford = mean(selected_HI);
    std_in_oxford = std(selected_HI);
    mu_out_oxford = mean(selected_Q);
    std_out_oxford = std(selected_Q);
    selected_HI   = (selected_HI - mu_in_oxford) ./ std_in_oxford;
    selected_Q   = (selected_Q - mu_out_oxford) ./ std_out_oxford;

	Estimation_base = polyval(MDl_base, selected_HI);
	error = selected_Q - Estimation_base;
	
	len = size(error, 1);
    Input_error = [Input_error; [Estimation_base, [1, 100, 1, 2].*ones(len, 4)]];
    Output_error = [Output_error; error];

    % CALCE
    i = 1;
    cellNameG = fieldnames(HI_CALCE);
    whole_size = size(HI_CALCE.(cellNameG{i,1}), 1);
	percentage = 0.05;
	num_extraction = ceil(whole_size*percentage);
	random_indices = sort(randperm(whole_size, num_extraction));

    HI_real = HI_CALCE.(cellNameG{i,1});
	Q_real = Q_CALCE.(cellNameG{i,1});
	
	selected_HI = HI_real(random_indices,:); 
	selected_Q = Q_real(random_indices, :);

    mu_in_CALCE = mean(selected_HI);
    std_in_CALCE = std(selected_HI);
    mu_out_CALCE = mean(selected_Q);
    std_out_CALCE = std(selected_Q);
    selected_HI   = (selected_HI - mu_in_CALCE) ./ std_in_CALCE;
    selected_Q   = (selected_Q - mu_out_CALCE) ./ std_out_CALCE;

	Estimation_base = polyval(MDl_base, selected_HI);
	error = selected_Q - Estimation_base;
	
	len = size(error, 1);
    Input_error = [Input_error; [Estimation_base, [1, 100, 1, 3].*ones(len, 4)]];
    Output_error = [Output_error; error];


    time_proposed_2 = tic;
    Mdl_error = fitrgp(Input_error, Output_error, 'KernelFunction', 'ardmatern32');
    time_proposed_2 = toc(time_proposed_2);
    disp(time_proposed_2+time_proposed_1)
    % MIT training results
    for i = 1:size(list, 1)
        cells = fieldnames(HI_MIT.(list{i,1}));
        whole_size = size(HI_MIT.(list{i,1}).(cells{1,1}).Results, 1);
        percentage = 0.01;
        num_extraction = ceil(whole_size*percentage);
        random_indices = sort(randperm(whole_size, num_extraction));
        selected_HI = mean(HI_MIT.(list{i,1}).(cells{1, 1}).Results(random_indices,[1,2,4]), 2);
        selected_Q = HI_MIT.(list{i,1}).(cells{1, 1}).Results(random_indices, end);

        selected_HI   = (selected_HI - mu_in_MIT) ./ std_in_MIT;
        selected_Q   = (selected_Q - mu_out_MIT) ./ std_out_MIT;
        len = size(selected_Q, 1);
        Estimation_base = polyval(MDl_base, selected_HI);
        est_error = predict(Mdl_error, [Estimation_base, [conditions(i,:), 1].*ones(len, 4)]);

        if seedIdx == 1
            Error.(list{i,1}).base_cap = Estimation_base* std_out_MIT + mu_out_MIT; %  * std_out_MIT + mu_out_MIT
            Error.(list{i,1}).error_real = (selected_Q - Estimation_base)* std_out_MIT;
            Error.(list{i,1}).error_est = est_error* std_out_MIT;
            Error.(list{i,1}).capacity_real = HI_MIT.(list{i,1}).(cells{1, 1}).Results(random_indices, end);
            Error.(list{i,1}).capacity_est = (est_error + Estimation_base)*std_out_MIT + mu_out_MIT;
            if i == 8
                save('.\data\MIT\Error_overview.mat', "Error")
                clearvars Error
            end
        end
    end
    % Oxford and CALCE training results
    % oxford
    if seedIdx == 1
        i = 1;
        cellNameG_oxford = fieldnames(HI_oxford);
        whole_size = size(HI_oxford.(cellNameG_oxford{i,1}), 1);
	    percentage = 0.1;
	    num_extraction = ceil(whole_size*percentage);
	    random_indices = sort(randperm(whole_size, num_extraction));
    
        HI_real = HI_oxford.(cellNameG_oxford{i,1})/1000;
	    Q_real = Q_oxford.(cellNameG_oxford{i,1})/1000;
	    
	    selected_HI = HI_real(random_indices,:); 
	    selected_Q = Q_real(random_indices, :);
        
        selected_HI   = (selected_HI - mu_in_oxford) ./ std_in_oxford;
        selected_Q   = (selected_Q - mu_out_oxford) ./ std_out_oxford;
            len = size(selected_Q, 1);

	    Estimation_base = polyval(MDl_base, selected_HI);
	    est_error = predict(Mdl_error, [Estimation_base, [1, 100, 1, 2].*ones(len, 4)]);
	    Error.base_cap = Estimation_base* std_out_oxford + mu_out_oxford; %  * std_out_MIT + mu_out_MIT
        Error.error_real = (selected_Q - Estimation_base)* std_out_oxford;
        Error.error_est = est_error* std_out_oxford;
        Error.capacity_real = Q_real(random_indices, :);
        Error.capacity_est = (est_error + Estimation_base)*std_out_oxford + mu_out_oxford;
        save('.\data\oxford\Error_overview.mat', "Error")
        clearvars Error

        % CALCE
        i = 1;
        cellNameG = fieldnames(HI_CALCE);
        whole_size = size(HI_CALCE.(cellNameG{i,1}), 1);
	    percentage = 0.05;
	    num_extraction = ceil(whole_size*percentage);
	    random_indices = sort(randperm(whole_size, num_extraction));
    
        HI_real = HI_CALCE.(cellNameG{i,1});
	    Q_real = Q_CALCE.(cellNameG{i,1});
	    
	    selected_HI = HI_real(random_indices,:); 
	    selected_Q = Q_real(random_indices, :);
   
        selected_HI   = (selected_HI - mu_in_CALCE) ./ std_in_CALCE;
        selected_Q   = (selected_Q - mu_out_CALCE) ./ std_out_CALCE;
            len = size(selected_Q, 1);

	    Estimation_base = polyval(MDl_base, selected_HI);
	    error = selected_Q - Estimation_base;
        est_error = predict(Mdl_error, [Estimation_base, [1, 100, 1, 3].*ones(len, 4)]);
	    Error.base_cap = Estimation_base* std_out_CALCE + mu_out_CALCE; %  * std_out_MIT + mu_out_MIT
        Error.error_real = (selected_Q - Estimation_base)* std_out_CALCE;
        Error.error_est = est_error* std_out_CALCE;
        Error.capacity_real = Q_real(random_indices, :);
        Error.capacity_est = (est_error + Estimation_base)*std_out_CALCE + mu_out_CALCE;
        save('.\data\CALCE\Error_overview.mat', "Error")
        clearvars Error
    end

    % test
    %% Verification on test data: other cells of all MCC fast-charging protocols
	MAPE_base = [];
	RMSE_base = [];
	MAPE_final = [];
	RMSE_final = [];
	MAPE = [];
	RMSE = [];
	for i = 1:size(list, 1)
		cells = fieldnames(HI_MIT.(list{i, 1}));
		for j = 2:size(cells, 1)
			HI = mean(HI_MIT.(list{i,1}).(cells{j, 1}).Results(:,[1,2,4]), 2);
            
            Q_real = HI_MIT.(list{i,1}).(cells{j, 1}).Results(:,end);

            HI   = (HI - mu_in_MIT) ./ std_in_MIT;
			Q_base = polyval(MDl_base, HI);
            E_base = Q_base - Q_real;
			RE_base = (Q_base - Q_real)./Q_real * 100;
			mape_base = mean(abs(E_base)./Q_real*100);
			rmse_base = sqrt(mean(E_base.^2));
			MAPE_base = [MAPE_base; mape_base];
			RMSE_base = [RMSE_base; rmse_base];
			len = size(Q_base, 1);
			Q_error = predict(Mdl_error, [Q_base, [conditions(i,:), 1].*ones(len, 4)]);

            Q_est = Q_base + Q_error;
            			Q_est = Q_est*std_out_MIT + mu_out_MIT;

            AE_final = Q_est - Q_real;
			RE_final = (Q_est - Q_real)./Q_real*100;
			mape_final = mape(Q_est, Q_real);
			rmse_final = rmse(Q_est, Q_real);
			% mape_final = mape(Q_est, Q_real);
			% rmse_final = rmse(Q_est, Q_real);
			MAPE_final = [MAPE_final; mape_final];
			RMSE_final = [RMSE_final;rmse_final];

            if i == 1 && j == 2
                max_mape = [mape_final, i, j-1];
                max_rmse = [rmse_final, i, j-1];
                min_mape = [mape_final, i, j-1];
                min_rmse = [rmse_final, i, j-1];
            end
            if max_mape(1) < mape_final
                max_mape = [mape_final, i, j-1];
            end
            if min_mape(1) > mape_final
                min_mape = [mape_final, i, j-1];
            end
            if max_rmse(1) < rmse_final
                max_rmse = [rmse_final, i, j-1];
            end
            if min_rmse(1) > rmse_final
                min_rmse = [rmse_final, i, j-1];
            end
                 
            Estimation_results_MIT.(seedName).proposed.(list{i,1}).(cells{j,1}).Results = ...
                [Q_real, Q_base, RE_base, Q_error, Q_est, AE_final];
          
		end
	end
	Estimation_results_MIT.(seedName).proposed.MAPE_base = MAPE_base;
	Estimation_results_MIT.(seedName).proposed.RMSE_base = RMSE_base;
	Estimation_results_MIT.(seedName).proposed.MAPE_final = MAPE_final;
	Estimation_results_MIT.(seedName).proposed.RMSE_final = RMSE_final;


    Estimation_results_MIT.(seedName).proposed.max_rmse = max_rmse;
    Estimation_results_MIT.(seedName).proposed.max_mape = max_mape;
    Estimation_results_MIT.(seedName).proposed.min_rmse = min_rmse;
    Estimation_results_MIT.(seedName).proposed.min_mape = min_mape;

    %% Verification on test data: other cells of all MCC fast-charging protocols
    MAPE_base = [];
	RMSE_base = [];
	MAPE_final = [];
	RMSE_final = [];
	for i = 1:8
		HI = HI_oxford.(cellNameG_oxford{i, 1})/1000;
		Q_real = Q_oxford.(cellNameG_oxford{i,1})/1000;

        HI   = (HI - mu_in_oxford) ./ std_in_oxford;

		Q_base = polyval(MDl_base, HI);
        E_base = Q_base - Q_real;
		RE_base = (Q_base - Q_real)./Q_real * 100;
		mape_base = mean(abs(E_base)./Q_real*100);
		rmse_base = sqrt(mean(E_base.^2));
		MAPE_base = [MAPE_base; mape_base];
		RMSE_base = [RMSE_base; rmse_base];
        
        len = size(Q_base, 1);
        Q_error = predict(Mdl_error, [Q_base, [1, 100, 1, 2].*ones(len, 4)]);
		Q_est = Q_base + Q_error;
                Q_est = Q_est*std_out_oxford + mu_out_oxford;

		RE_final = (Q_est - Q_real)./Q_real*100;
        AE_final = Q_est - Q_real;
		mape_final = mean(abs(AE_final)./Q_real*100);
					rmse_final = rmse(Q_est, Q_real);
		MAPE_final = [MAPE_final; mape_final];
		RMSE_final = [RMSE_final;rmse_final];
		Estimation_results_oxford.(seedName).proposed.(cellNameG_oxford{i, 1}).Results = ...
            [Q_real, Q_base, RE_base, Q_error, Q_est, AE_final];
	end
	Estimation_results_oxford.(seedName).proposed.MAPE_base = MAPE_base;
	Estimation_results_oxford.(seedName).proposed.RMSE_base = RMSE_base;
	Estimation_results_oxford.(seedName).proposed.MAPE_final = MAPE_final;
	Estimation_results_oxford.(seedName).proposed.RMSE_final = RMSE_final;

    %%
    MAPE_base = [];
	RMSE_base = [];
	MAPE_final = [];
	RMSE_final = [];

	for i = 1:4
		HI = HI_CALCE.(cellNameG{i, 1});
		Q_real = Q_CALCE.(cellNameG{i,1});
                HI   = (HI - mu_in_CALCE) ./ std_in_CALCE;

		Q_base = polyval(MDl_base, HI);
        E_base = Q_base - Q_real;
		RE_base = (Q_base - Q_real)./Q_real * 100;
		mape_base = mean(abs(E_base)./Q_real*100);
		rmse_base = sqrt(mean(E_base.^2));
		MAPE_base = [MAPE_base; mape_base];
		RMSE_base = [RMSE_base; rmse_base];

       len = size(Q_base, 1);
        Q_error = predict(Mdl_error, [Q_base, [1, 100, 1, 3].*ones(len, 4)]);

		Q_est = Q_base + Q_error;
                        Q_est = Q_est*std_out_CALCE + mu_out_CALCE;

		RE_final = (Q_est - Q_real)./Q_real*100;
        AE_final = Q_est - Q_real;
		mape_final = mean(abs(AE_final)./Q_real*100);
					rmse_final = rmse(Q_est, Q_real);
		MAPE_final = [MAPE_final; mape_final];
		RMSE_final = [RMSE_final;rmse_final];
		Estimation_results_CALCE.(seedName).proposed.(cellNameG{i, 1}).Results = ...
            [Q_real, Q_base, RE_base, Q_error, Q_est, AE_final];
	end
	Estimation_results_CALCE.(seedName).proposed.MAPE_base = MAPE_base;
	Estimation_results_CALCE.(seedName).proposed.RMSE_base = RMSE_base;
	Estimation_results_CALCE.(seedName).proposed.MAPE_final = MAPE_final;
	Estimation_results_CALCE.(seedName).proposed.RMSE_final = RMSE_final;
end
save(".\data\MIT\Estimation_results.mat", "Estimation_results_MIT")
save(".\data\oxford\Estimation_results.mat", "Estimation_results_oxford")
save(".\data\CALCE\Estimation_results.mat", "Estimation_results_CALCE")
