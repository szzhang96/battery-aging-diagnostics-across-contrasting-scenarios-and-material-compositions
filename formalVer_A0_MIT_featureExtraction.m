% Data preprocessing, HI extraction and correlation analysis

clear
clc

%% Load batch data
load('2018-04-12_batchdata_updated_struct_errorcorrect.mat')
for i= 1:size(batch, 2)
    batch(i).id= i;
end
[~,index]= sortrows({batch.policy}.');% list all MCC fast-charging protocols in ascend oder;
batch= batch(index);

%% Divide batch data according to MCC fast-charging protocols and save them
Abnormal_cell_cycle=[];% record abnormal data cycle
policy_compare= batch(1).policy;
for i= 1:size(batch, 2)
    if strcmp(batch(i).policy,policy_compare)==1
        allData.(['p',policy_compare]).(['cell',num2str(batch(i).id)]) = batch(i); % save reconstructed time-voltage data & capacity_max
    else
        policy_compare = batch(i).policy;
        allData.(['p',policy_compare]).(['cell',num2str(batch(i).id)]) = batch(i); % save reconstructed time-voltage data & capacity_max
    end
end

%% Extract all potential HIs
list = fieldnames(allData);  % read all MCC fast-charging protocols
for i = 1:size(list, 1)
    cells = fieldnames(allData.(list{i,1})); % read all cells under the single MCC fast-charging protocol
    for j = 1:size(cells, 1)
        cycles = size(allData.(list{i,1}).(cells{j, 1}).cycles, 2); % read cycle life
        HI_total = []; % record all potential HIs
        Qmax_total = []; % read cell capacity
        for k = 1:cycles
            if i == 1 && j == 1 && k == 551
                continue
            end
            currentCycleData = allData.(list{i,1}).(cells{j,1}).cycles(k);
            HI_Start = find(currentCycleData.I == 0,1); % locate the relaxation stage
            if i == 1 && j == 1 && k == 1
                U = allData.(list{i, 1}).(cells{j, 1}).cycles(k).V;
                I = allData.(list{i, 1}).(cells{j, 1}).cycles(k).I;
                t = allData.(list{i, 1}).(cells{j, 1}).cycles(k).t;
                save('HI_extraction_overview_MIT.mat',"t","I","U","HI_Start")
            end
            HI1 = currentCycleData.V(HI_Start); % maximum voltage
            HI2 = currentCycleData.V(HI_Start + 10); % minimum voltage
            p = polyfit(1:11, currentCycleData.V(HI_Start : HI_Start+10), 1);
            HI3 = p(2);
            HI4 = mean(currentCycleData.V(HI_Start : HI_Start+10));
            HI5 = var(currentCycleData.V(HI_Start : HI_Start+10));
            HI6 = 100*std(currentCycleData.V(HI_Start : HI_Start+10))/HI4;
            HI7 = skewness(currentCycleData.V(HI_Start : HI_Start+10), 0);
            HI8 = kurtosis(currentCycleData.V(HI_Start : HI_Start+10), 0);
            Qmax = currentCycleData.Qc(end); % cell capacity
            HI_total = [HI_total; HI1, HI2, HI3, HI4, HI5, HI6, HI7, HI8];
            Qmax_total = [Qmax_total; Qmax];
        end
        HI_extraction.(list{i,1}).(cells{j, 1}).Results = [HI_total, Qmax_total];
        HI_extraction.(list{i,1}).(cells{j, 1}).EOL = size(HI_total, 1);
    end
end
save('HI.mat',"HI_extraction") % save all potentials HIs and the recorded cell capacity

%% Correlation analysis
list = fieldnames(HI_extraction); % read all MCC fast-charging protocols
correlation_total = []; % record correlation coefficients
for i = 1:size(list, 1)
    cells = fieldnames(HI_extraction.(list{i, 1})); % read all cells under the single MCC fast-charging protocol
    correlation_cells = [];
    for j = 1:size(cells, 1)
        correlation_HI = [];
        for k = 1:8
            HI = HI_extraction.(list{i,1}).(cells{j, 1}).Results(:, k);
            Qmax = HI_extraction.(list{i,1}).(cells{j, 1}).Results(:, end);
            correlation_HI = [correlation_HI; corr(HI, Qmax)];
        end
        correlation_cells = [correlation_cells, correlation_HI];
    end
    correlation_total = [correlation_total, correlation_cells];
end
save('Correlation_analysis.mat', 'correlation_total')
