clear all;
clc;

SolarProfile = xlsread('Solar_Profile_2hr_1sec.xlsx');
[row,col] = size(SolarProfile);
PV = SolarProfile(:,2:5);
slope = zeros(row,col-1);

for j = 1:col-1    
    for t = 1:row
        if t == 1
            slope(t,j) = 1;
        else
            slope(t,j) =  PV(t,j)/PV(t-1,j);
        end
    end
end

slope = [slope; ones(row,1)*slope(row,:)];
xlswrite('SolarPrediction_2hr_1sec.xlsx',slope);