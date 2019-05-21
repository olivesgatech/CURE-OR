%% BGS/DEVS/PERSPS ALL SEPARATE & ORIGINAL IMAGES
%  5*5*5 * (cLevelN + 1): 5 levels -> 750 columns 
IQAs = ["psnr", "ssim", "mslUNIQUE"];
methods  = ["Spearman"];
bgs = 1:5; devs = 1:5; persps = 1:5;
objs = [99, 68, 5, 71, 40, 12, 23, 61, 94, 69];
numObj = length(objs);
cLevelN = 5;
maxIQAs = [55, 1, 1, 1];

if ~exist('Results/IQA/Data', 'dir')
    mkdir Results/IQA/Data
end

%% Load each object's IQA results by bgs/devs/persps/cLevels (lev 0-3)
cLevelN = 5; % lev 1-5

IQAs_555 = struct;
for i = 1:length(IQAs)
    IQAs_555.(IQAs(i)) = horzcat(repmat(maxIQAs(i),8,5*5*5), zeros(8,5*5*5*(cLevelN)));
end

for obj = objs
    for bg = bgs
        for dev = devs
            for persp = persps
                filename = sprintf('IQA/Result/%03d/%d_%d_%d_%03d.mat',obj,bg,dev,persp,obj);
                load(filename);
               
                ind = (bg - 1)*(5^2) + (dev - 1)*5 + (persp - 1) + 1;
                loc = 5*5*5 + ind : 5*5*5 : 5*5*5*(cLevelN + 1);  
                for IQA = IQAs
                    IQAs_555.(IQA)(:,loc) = IQAs_555.(IQA)(:,loc) + IQA_vals.(IQA);
                end
            end
        end
    end
end

for IQA = IQAs
    IQAs_555.(IQA)(:,5*5*5 + 1:end) = IQAs_555.(IQA)(:,5*5*5 + 1:end) ./ numObj;
end

save('Results/IQA/Data/IQAs_555_10obj.mat','IQAs_555');


%% Load IQA results & performance results
load(sprintf('Results/IQA/Data/IQAs_555_10obj.mat'));
perf1 = csvread('Results/IQA/Data/AWS_color_N_5_obj_10.csv',1,1);
perf2 = csvread('Results/IQA/Data/Azure_color_N_5_obj_10.csv',1,1);

perf = cat(3, perf1, perf2);
perf = mean(perf, 3);
csvwrite('Results/IQA/Data/BothApps_color_N_5_obj_10.csv', perf);
            
corr_coefs = struct;

%% Bgs: 1-3, Devs: 1-5, Persps: Averaged
levStart = 1 : 5*5*5 : 5*5*5*(cLevelN + 1);
ind = [];
for lev = levStart
    ind = horzcat(ind, lev : lev + (3*5*5 - 1)); % first 3 backgrounds
end

n = 5; % average out every n elements
for m = methods
    for i = 1:length(IQAs)
        for cType = 1:8
            IQA_tmp = IQAs_555.(IQAs(i))(cType, ind);
            perf_tmp = perf(cType, ind);
            
            %% Mean of every n elements (for averaging perspectives)
            IQA_tmp_avg = arrayfun(@(i) mean(IQA_tmp(i:i+n-1)), 1:5:length(IQA_tmp)-n+1);
            perf_tmp_avg = arrayfun(@(i) mean(perf_tmp(i:i+n-1)), 1:5:length(perf_tmp)-n+1);
            
            corr_coefs.(m)(cType, i) = corr(IQA_tmp_avg', perf_tmp_avg', 'Type', m);
        end
        
        IQA_tmp = mean(IQAs_555.(IQAs(i))(:, ind));
        perf_tmp = mean(perf(:, ind));
        
        IQA_tmp_avg = arrayfun(@(i) mean(IQA_tmp(i:i+n-1)), 1:5:length(IQA_tmp)-n+1);
        perf_tmp_avg = arrayfun(@(i) mean(perf_tmp(i:i+n-1)), 1:5:length(perf_tmp)-n+1);

%         corr_coefs.(m)(9,i) = corr(IQA_tmp_avg', perf_tmp_avg', 'Type', m);
    end
        
    disp(m)
    disp([IQAs])
    disp(round(corr_coefs.(m), 3))
    
end

csvwrite('Results/IQA/Data/BothApps_color_N_5_obj_10_corr_coefs.csv',corr_coefs(1).Spearman);


%% Bgs: 1-3, Devs: 1-5, Persps: Averaged, Levels: 0-5, Object: 10 common objects
cLevelN = 5;
levStart = 1 : 5*5*5 : 5*5*5*(cLevelN + 1);
ind = [];
for lev = levStart
    ind = horzcat(ind, lev : lev + (3*5*5 - 1)); % first 3 backgrounds
end

n = 5; % average out every n elements: perspectives
bgsN = 3; % Background 1-3 only

% m: markers, c: colors
m = {'d', 'o', 's', '^', '+', 'h'};
c = {'r', 'g', 'b', 'm', 'c', 'y'};

xlabels = ["PSNR", "SSIM", "UNIQUE"];
mkSize = 110; lgdSize = 22; lgdMkSize = 15; 
fontSize = 13; yFontSize = 11; axisFontSize =11;

path = sprintf('Results/IQA/Data/BothApps_color_N_5_obj_10.csv');
perf = csvread(path,1,1);

IQA_concat = zeros(cLevelN + 1, length(IQAs)*n*bgsN); 
perf_concat = zeros(cLevelN + 1, length(IQAs)*n*bgsN);

for i = 1:length(IQAs)

    IQA_tmp = mean(IQAs_555.(IQAs(i))(:, ind));
    perf_tmp = mean(perf(:, ind));

    % Resize Level 5: make it NaN s.t. not taking into account for
    % mean calculation
    IQA_tmp(1,5*5*5*5+1:5*5*5*6) = NaN;
    perf_tmp(1,5*5*5*5+1:5*5*5*6) = NaN;

    IQA_tmp_avg = arrayfun(@(i) mean(IQA_tmp(i:i+n-1)), 1:5:length(IQA_tmp)-n+1); 
    IQA_tmp_avg(IQA_tmp_avg == maxIQAs(i)) = max(IQA_tmp_avg(16:end));
    perf_tmp_avg = arrayfun(@(i) mean(perf_tmp(i:i+n-1)), 1:5:length(perf_tmp)-n+1);

    % Saving IQA & perf as csv for python plotting
    concatCol = (i - 1)*n*bgsN + 1 : i*n*bgsN;
    for row = 1:cLevelN + 1
        origCol = (row - 1)*n*bgsN + 1 : row*n*bgsN;
        IQA_concat(row, concatCol) = IQA_tmp_avg(origCol);
        perf_concat(row, concatCol) = perf_tmp_avg(origCol);
    end

csvwrite('Results/IQA/Data/IQA_concat_allLev.csv', IQA_concat);
csvwrite('Results/IQA/Data/Perf_concat_allLev.csv', perf_concat);

end
