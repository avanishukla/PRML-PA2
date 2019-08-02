
clc;
clearvars;
mydir = 'D:\prml\Dataset_2(b)_Varying_Length\2b\';

%class1--------------------------------------------------------------------------------
class1_path = strcat(mydir,'090.gorilla\');
foldername = fullfile(class1_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
%list = list(1:10);
class1_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[64 Inf]);
           fileread = fileread';
        class1_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class1 = floor(0.7*size(class1_cell,1));
size_validation_class1 = floor(0.1*size(class1_cell,1));
size_test_class1 = size(class1_cell,1) - size_train_class1 - size_validation_class1;
train1_cell = class1_cell(1:size_train_class1,1);
validation1_cell = class1_cell(size_train_class1+1:size_train_class1+size_validation_class1,1);
test1_cell = class1_cell(size_train_class1+size_validation_class1+1:size_train_class1+size_validation_class1+size_test_class1,1);

train_data1 = cell2mat(train1_cell);
val_data1 = cell2mat(validation1_cell);
test_data1 = cell2mat(test1_cell);

%class2--------------------------------------------------------------------------------
class2_path = strcat(mydir,'105.horse\');
foldername = fullfile(class2_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
%list = list(1:10);
class2_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[64 Inf]);
        fileread = fileread';
        class2_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class2 = floor(0.7*size(class2_cell,1));
size_validation_class2 = floor(0.1*size(class2_cell,1));
size_test_class2 = size(class2_cell,1) - size_train_class2 - size_validation_class2;
train2_cell = class2_cell(1:size_train_class2,1);
validation2_cell = class2_cell(size_train_class2+1:size_train_class2+size_validation_class2,1);
test2_cell = class2_cell(size_train_class2+size_validation_class2+1:size_train_class2+size_validation_class2+size_test_class2,1);

train_data2 = cell2mat(train2_cell);
val_data2 = cell2mat(validation2_cell);
test_data2 = cell2mat(test2_cell);

%class3--------------------------------------------------------------------------------
class3_path = strcat(mydir,'158.penguin\');
foldername = fullfile(class3_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
%list = list(1:10);
class3_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[64 Inf]);
        fileread = fileread';
        class3_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class3 = floor(0.7*size(class3_cell,1));
size_validation_class3 = floor(0.1*size(class3_cell,1));
size_test_class3 = size(class3_cell,1) - size_train_class3 - size_validation_class3;
train3_cell = class3_cell(1:size_train_class3,1);
validation3_cell = class3_cell(size_train_class3+1:size_train_class3+size_validation_class3,1);
test3_cell = class3_cell(size_train_class3+size_validation_class3+1:size_train_class3+size_validation_class3+size_test_class3,1);

train_data3 = cell2mat(train3_cell);
val_data3 = cell2mat(validation3_cell);
test_data3 = cell2mat(test3_cell);

%class4--------------------------------------------------------------------------------
class4_path = strcat(mydir,'232.t-shirt\');
foldername = fullfile(class4_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
%list = list(1:10);
class4_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[64 Inf]);
        fileread = fileread';
        class4_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class4 = floor(0.7*size(class4_cell,1));
size_validation_class4 = floor(0.1*size(class4_cell,1));
size_test_class4 = size(class4_cell,1) - size_train_class4 - size_validation_class4;
train4_cell = class4_cell(1:size_train_class4,1);
validation4_cell = class4_cell(size_train_class4+1:size_train_class4+size_validation_class4,1);
test4_cell = class4_cell(size_train_class4+size_validation_class4+1:size_train_class4+size_validation_class4+size_test_class4,1);

train_data4 = cell2mat(train4_cell);
val_data4 = cell2mat(validation4_cell);
test_data4 = cell2mat(test4_cell);

%class5--------------------------------------------------------------------------------
class5_path = strcat(mydir,'251.airplanes-101\');
foldername = fullfile(class5_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
%list = list(1:10);
class5_cell = cell(numel(list),1);
if(~isempty(list))
   for j = 1: numel(list)
        fileID = fopen(fullfile(list(j).folder,list(j).name),'r');
        fileread = fscanf(fileID,'%f',[64 Inf]);
        fileread = fileread';
        class5_cell{j,1} = fileread;
        fclose(fileID);
   end
end

size_train_class5 = floor(0.7*size(class5_cell,1));
size_validation_class5 = floor(0.1*size(class5_cell,1));
size_test_class5 = size(class5_cell,1) - size_train_class5 - size_validation_class5;
train5_cell = class5_cell(1:size_train_class5,1);
validation5_cell = class5_cell(size_train_class5+1:size_train_class5+size_validation_class5,1);
test5_cell = class5_cell(size_train_class5+size_validation_class5+1:size_train_class5+size_validation_class5+size_test_class5,1);

train_data5 = cell2mat(train5_cell);
val_data5 = cell2mat(validation5_cell);
test_data5 = cell2mat(test5_cell);

k = 7;
%train class1---------------------------------------------------------------------------------------
train_data = [train1_cell;train2_cell;train3_cell;train4_cell;train5_cell];
Actual_train = [ones(size(train2_cell,1),1);2.*ones(size(train2_cell,1),1);3.*ones(size(train3_cell,1),1);4.*ones(size(train4_cell,1),1);5.*ones(size(train5_cell,1),1)];
predicted_train = zeros(size(train_data,1),1);
train_error = 0;

for i = 1:size(train_data,1)
    dist = zeros(5,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(train_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data1(j,z)-train_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 1;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(train_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data2(j,z)-train_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data3,1),2);
    for j = 1:size(train_data3,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(train_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data3(j,z)-train_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 3;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(3,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data4,1),2);
    for j = 1:size(train_data4,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(train_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data4(j,z)-train_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(4,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data5,1),2);
    for j = 1:size(train_data5,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(train_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data5(j,z)-train_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 5;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(5,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_train(i) = dist(index,2);
    
    if Actual_train(i) ~= predicted_train(i)
        train_error = train_error+1;
    end
end

%validation---------------------------------------------------------------------------------------
val_data = [validation1_cell;validation2_cell;validation3_cell;validation4_cell;validation5_cell];
Actual_val = [ones(size(validation1_cell,1),1);2.*ones(size(validation2_cell,1),1);3.*ones(size(validation3_cell,1),1);4.*ones(size(validation4_cell,1),1);5.*ones(size(validation5_cell,1),1)];
predicted_val = zeros(size(val_data,1),1);
val_error = 0;

for i = 1:size(val_data,1)
    dist = zeros(5,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(val_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data1(j,z)-val_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 1;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(val_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data2(j,z)-val_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data3,1),2);
    for j = 1:size(train_data3,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(val_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data3(j,z)-val_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 3;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(3,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data4,1),2);
    for j = 1:size(train_data4,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(val_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data4(j,z)-val_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(4,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data5,1),2);
    for j = 1:size(train_data5,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(val_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data5(j,z)-val_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 5;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(5,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_val(i) = dist(index,2);
    
    if Actual_val(i) ~= predicted_val(i)
        val_error = val_error+1;
    end
end

%test---------------------------------------------------------------------------------------
test_data = [test1_cell;test2_cell;test3_cell;test4_cell;test5_cell];
Actual_test = [ones(size(test1_cell,1),1);2.*ones(size(test2_cell,1),1);3.*ones(size(test3_cell,1),1);4.*ones(size(test4_cell,1),1);5.*ones(size(test5_cell,1),1)];
predicted_test = zeros(size(test_data,1),1);
test_error = 0;
for i = 1:size(test_data,1)
    dist = zeros(5,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(test_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data1(j,z)-test_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 1;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(test_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data2(j,z)-test_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data3,1),2);
    for j = 1:size(train_data3,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(test_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data3(j,z)-test_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 3;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(3,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data4,1),2);
    for j = 1:size(train_data4,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(test_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data4(j,z)-test_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(4,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data5,1),2);
    for j = 1:size(train_data5,1)
        dist_matrix(j,1) = 1;
        for o = 1:size(test_data{i,1},1)
            radius = 0;
        for z=1:64
            radius = radius + (train_data5(j,z)-test_data{i,1}(o,z))^2;
        end
            dist_matrix(j,1) = dist_matrix(j,1) * radius;
        end
        dist_matrix(j,2) = 5;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(5,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_test(i) = dist(index,2);
    
    if Actual_test(i) ~= predicted_test(i)
        test_error = test_error+1;
    end
end

acc_train = (size(train_data,1) - train_error)*100/size(train_data,1);
acc_val = (size(val_data,1) - val_error)*100/size(val_data,1);
acc_test = (size(test_data,1) - test_error)*100/size(test_data,1);

confusion_mat_train = zeros(5,5);

for i=1:size(predicted_train,1)
    row = Actual_train(i);
    col = predicted_train(i);
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_test = zeros(5,5);
for i=1:size(predicted_test,1)
    row = Actual_test(i);
    col = predicted_test(i);
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end