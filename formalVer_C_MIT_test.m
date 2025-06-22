% Test

clear
clc

load HI.mat
load Mdl_base.mat
load tf_sourceMdl_opt.mat
list = fieldnames(HI_extraction); %read all MCC fast-charging protocols
conditions = [3.7, 31, 5.9; 4.9 80, 4.9; 5, 67, 4; ...
    5.3, 54, 4; 5.6, 19, 4.6; 5.6, 36, 4.3; 5.9, 15, 4.6; ...
    5.9, 60, 3.1]; % typical parameters of MCC fast-charging protocols
for seedIdx = 1:10
	seedName = strcat('seed_', num2str(seedIdx));
	load(strcat('Error_model_',seedName,'.mat'))
	load(strcat('tf_mdls_',seedName,'.mat'))
	
	%% Verification on test data: other cells of all MCC fast-charging protocols
	MAPE_base = [];
	RMSE_base = [];
	MAPE_final = [];
	RMSE_final = [];
	MAPE = [];
	RMSE = [];
	for i = 1:size(list, 1)
		cells = fieldnames(HI_extraction.(list{i, 1}));
		for j = 2:size(cells, 1)
			HI = mean(HI_extraction.(list{i,1}).(cells{j, 1}).Results(:,[1,2,4]), 2);
			
            
            
            Q_real = HI_extraction.(list{i,1}).(cells{j, 1}).Results(:,end);
			Q_base = polyval(MDl_base, HI);
            E_base = Q_base - Q_real;
			RE_base = (Q_base - Q_real)./Q_real * 100;
			mape_base = mean(abs(E_base)./Q_real*100);
			rmse_base = sqrt(mean(E_base.^2));
			MAPE_base = [MAPE_base; mape_base];
			RMSE_base = [RMSE_base; rmse_base];
			len = size(Q_base, 1);
			Q_error = predict(Mdl_error, [Q_base, conditions(i,:).*ones(len, 3)]);
			
            Q_est = Q_base + Q_error;
            E_final = Q_est - Q_real;
			RE_final = (Q_est - Q_real)./Q_real*100;
			mape_final = mean(abs(E_final)./Q_real*100);
			rmse_final = rms(E_final);
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
                 
            Estimation_results.(seedName).proposed.(list{i,1}).(cells{j,1}).Results = ...
                [Q_real, Q_base, RE_base, Q_error, Q_est, abs(E_final)];
            
			HI_normalized = (HI-mu_input)/sigma_input;
			Q_est = predict(netWhole(i), HI_normalized')*sigma_output+mu_output;
			E = Q_est' - Q_real;
            RE = (Q_est'-Q_real)./Q_real * 100;
			MAPE = [MAPE;  mean(abs(E)./Q_real*100)];
			RMSE = [RMSE; sqrt(mean(E.^2))];
			Estimation_results.(seedName).transfer.(list{i,1}).(cells{j,1}).Results = ...
            [Q_real, Q_est', abs(E)];	
		end
	end
	Estimation_results.(seedName).proposed.MAPE_base = MAPE_base;
	Estimation_results.(seedName).proposed.RMSE_base = RMSE_base;
	Estimation_results.(seedName).proposed.MAPE_final = MAPE_final;
	Estimation_results.(seedName).proposed.RMSE_final = RMSE_final;
	Estimation_results.(seedName).transfer.MAPE_final = MAPE;
	Estimation_results.(seedName).transfer.RMSE_final = RMSE;

    Estimation_results.(seedName).proposed.max_rmse = max_rmse;
    Estimation_results.(seedName).proposed.max_mape = max_mape;
    Estimation_results.(seedName).proposed.min_rmse = min_rmse;
    Estimation_results.(seedName).proposed.min_mape = min_mape;
   
end
save('Estimation_results.mat',"Estimation_results")
		