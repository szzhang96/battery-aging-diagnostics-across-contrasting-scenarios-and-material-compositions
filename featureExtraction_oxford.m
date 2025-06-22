% Feature extraction

clc
clear
load D:\STUDY\BATTERY_DATASET\Oxford\Oxford_Battery_Degradation_Dataset_1.mat

vars = who;
HI_all_2 = [];
Q_all_2 = [];
HI_all = [];
Q_all = [];
for i = 1:length(vars)
    varName = vars{i};
    cycName = fieldnames(eval(varName));
	HI = [];
	Q = [];
	for j = 1:size(cycName, 1)
        data = eval([varName '.' cycName{j,1}]);
		charData = data.C1ch;
		v = charData.v;
        q = charData.q;
		sv = 3.8;
		ev = 4;
		sp = find(v>=sv, 1);
		ep = find(v>=ev, 1);
        if i ==1 && j == 1
            save('HI_extraction_overview_oxford.mat',"v","sp","ep")
        end
        if i == 5 && j == size(cycName, 1)
            continue
        end
        HI = [HI; q(ep)-q(sp)];
		Q = [Q; q(end)];

        cName = cycName{j,1};
        allData.(vars{i,1}).(cName).char = v;
        allData.(vars{i,1}).(cName).dis = data.C1dc.v;
        
    end

	HI_all = [HI_all; HI];
	Q_all = [Q_all; Q];
	HI_all_2.(vars{i,1}) = HI;
	Q_all_2.(vars{i,1}) = Q;
	disp(corr(Q, HI))
    allData.(vars{i,1}).Q = Q;

end
save('HI.mat', "HI_all", "HI_all_2", "Q_all", "Q_all_2")
save('oxford_allbatteries.mat',"allData")
