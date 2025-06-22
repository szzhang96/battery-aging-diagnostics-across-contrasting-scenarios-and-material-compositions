% Feature extraction

clc
clear
HI_all_2 = [];
Q_all_2 = [];
HI_all = [];
Q_all = [];

cellNameG = {'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'};
for i = 1:4
    load(strcat('D:\STUDY\BATTERY_DATASET\CALCE\',cellNameG{1,i},'.mat'))
    HI = [];
    Q = [];
    for j = 1:size(All_Char_Data, 1)
        charData = All_Char_Data(j).ConstantCurrent;
        vol = charData.Voltage_V_;
        cur = charData.Current_A_;
        t = charData.Date_Time;
        t = [0; seconds(diff(t))];
        if vol(1) > 3.9
            continue
        end
        q = cumsum(cur.*t)/3600;
        if ~isempty(find(t>60,1))
            continue
        end
        sv = 3.8;
        ev = 4;
        sp = find(vol>=sv, 1);
        ep = find(vol>=ev, 1);

        vol = All_Char_Data(j).data.Voltage_V_;
        cur = All_Char_Data(j).data.Current_A_;
        t = All_Char_Data(j).data.Test_Time_s_;

        if i == 1 && j == 1
            save('HI_extraction_overview_CALCE.mat',"vol","cur","t","sp","ep")
        end
        HI = [HI; q(ep)-q(sp)];
        Q = [Q; q(end)];
        if i == 1
            data.(strcat('cyc',num2str(j))).t = t;
            data.(strcat('cyc',num2str(j))).V = vol;
            data.(strcat('cyc',num2str(j))).I = cur;
        end
    end
    if i == 1
        save('vol_data.mat',"data")
    end
    HI_all = [HI_all; HI];
    Q_all = [Q_all; Q];
    HI_all_2.(cellNameG{1, i}) = HI;
    Q_all_2.(cellNameG{1, i}) = Q;
    disp(corr(Q, HI))
end
save('HI.mat', "HI_all", "HI_all_2", "Q_all", "Q_all_2")
