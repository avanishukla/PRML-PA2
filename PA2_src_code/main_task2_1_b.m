clc;
clearvars;
mydir = 'D:\prml\Dataset_2(b)_Varying_Length\2b\';

k1 = 2;
k2 = 2;
k3 = 2;
k4 = 2;
k5 = 2;

%class1--------------------------------------------------------------------------------
class1_path = strcat(mydir,'090.gorilla\');
foldername = fullfile(class1_path);
pth = genpath(foldername);
pathTest = regexp([pth ';'],'(.*?);','tokens');
list = dir([pathTest{1,1}{:} '\*.txt']);
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

theta = cell(5,3);
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

%train class1--------------------------------------------------------------------------------------
A = train_data1;
total_data = size(A,1);

[index,mean_value] = kmeans(A,k1);

data_per_cluster = ones(k1,1);
data = cell(1,k1);

for i=1:total_data
    tmp = data{1,(index(i))};
    tmp(data_per_cluster(index(i)),:) = A(i,:);
    data{1,(index(i))} = tmp;
    data_per_cluster(index(i)) = data_per_cluster(index(i)) + 1;
end

cov = cell(1,k1);
W = zeros(k1,1);
for i=1 : k1
    cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
    cov{1,i} = diag(diag(cov{1,i}));
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1: size_train_class1
        p_given_theta = 1;
        for z =  1: size(train1_cell{i,1},1)
            temp = 0;
            for j = 1 : k1
                temp = temp + W(j)*exp(-0.5*(train1_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train1_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
end

while abs(L_old - L_new)>10
    L_old = L_new;
    gama = zeros(total_data,k1);
    for i= 1 : total_data
        for j= 1: k1
            term(j) = W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        for j= 1: k1
            gama(i,j) = term(j)/sum(term);
        end
    end
    
    new_mean = gama' * A;
    for i= 1 : k1
        new_mean(i,:) = new_mean(i,:)/sum(gama(:,i)); 
    end
    new_cov = cell(1,k1); 
    for i= 1 : k1
        tmp_cov = zeros(64,64);
        for z = 1 : size(A,1)
            tmp_cov = tmp_cov + gama(z,i)*(A(z,:) - new_mean(i,:))'*(A(z,:) - new_mean(i,:));
        end
        new_cov{1,i} = tmp_cov/sum(gama(:,i)); 
        new_cov{1,i} = diag(diag(new_cov{1,i}));
    end
    new_W = zeros(k1,1);
    for i= 1 : k1
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    
    L_new = 0;
    
    for i= 1: size_train_class1
        p_given_theta = 1;
        for z =  1: size(train1_cell{i,1},1)
            temp = 0;
            for j = 1 : k1
                temp = temp + new_W(j)*exp(-0.5*(train1_cell{i,:}(z,:)-new_mean(j,:))*inv(new_cov{1,j})*(train1_cell{i,:}(z,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{1,1} = mean_value;
theta{1,2} = cov;
theta{1,3} = W;

%train class2--------------------------------------------------------------------------------------
clearvars data data_per_cluster gama term;
A = train_data2;
total_data = size(A,1);

[index,mean_value] = kmeans(A,k2);

data_per_cluster = ones(k2,1);
data = cell(1,k2);

for i=1:total_data
    tmp = data{1,(index(i))};
    tmp(data_per_cluster(index(i)),:) = A(i,:);
    data{1,(index(i))} = tmp;
    data_per_cluster(index(i)) = data_per_cluster(index(i)) + 1;
end

cov = cell(1,k2);
W = zeros(k2,1);
for i=1 : k2
    cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
    cov{1,i} = diag(diag(cov{1,i}));
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1: size_train_class2
        p_given_theta = 1;
        for z =  1: size(train2_cell{i,1},1)
            temp = 0;
            for j = 1 : k2
                temp = temp + W(j)*exp(-0.5*(train2_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train2_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
end

while abs(L_old - L_new)>10
    L_old = L_new;
    gama = zeros(total_data,k2);
    for i= 1 : total_data
        for j= 1: k2
            term(j) = W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        for j= 1: k2
            gama(i,j) = term(j)/sum(term);
        end
    end
    
    new_mean = gama' * A;
    for i= 1 : k2
        new_mean(i,:) = new_mean(i,:)/sum(gama(:,i)); 
    end
    new_cov = cell(1,k2); 
    for i= 1 : k2
        tmp_cov = zeros(64,64);
        for z = 1 : size(A,1)
            tmp_cov = tmp_cov + gama(z,i)*(A(z,:) - new_mean(i,:))'*(A(z,:) - new_mean(i,:));
        end
        new_cov{1,i} = tmp_cov/sum(gama(:,i)); 
        new_cov{1,i} = diag(diag(new_cov{1,i}));
    end
    new_W = zeros(k2,1);
    for i= 1 : k2
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    L_new = 0;
    for i= 1: size_train_class2
        p_given_theta = 1;
        for z =  1: size(train2_cell{i,1},1)
            temp = 0;
            for j = 1 : k2
                temp = temp + new_W(j)*exp(-0.5*(train2_cell{i,:}(z,:)-new_mean(j,:))*inv(new_cov{1,j})*(train2_cell{i,:}(z,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{2,1} = mean_value;
theta{2,2} = cov;
theta{2,3} = W;

%train class3--------------------------------------------------------------------------------------
clearvars data data_per_cluster gama term;
A = train_data3;
total_data = size(A,1);

[index,mean_value] = kmeans(A,k3);

data_per_cluster = ones(k3,1);
data = cell(1,k3);

for i=1:total_data
    tmp = data{1,(index(i))};
    tmp(data_per_cluster(index(i)),:) = A(i,:);
    data{1,(index(i))} = tmp;
    data_per_cluster(index(i)) = data_per_cluster(index(i)) + 1;
end

cov = cell(1,k3);
W = zeros(k3,1);
for i=1 : k3
    cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
    cov{1,i} = diag(diag(cov{1,i}));
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1: size_train_class3
        p_given_theta = 1;
        for z =  1: size(train3_cell{i,1},1)
            temp = 0;
            for j = 1 : k3
                temp = temp + W(j)*exp(-0.5*(train3_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train3_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
end

while abs(L_old - L_new)>10
    L_old = L_new;
    gama = zeros(total_data,k3);
    for i= 1 : total_data
        for j= 1: k3
            term(j) = W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        for j= 1: k3
            gama(i,j) = term(j)/sum(term);
        end
    end
    
    new_mean = gama' * A;
    for i= 1 : k3
        new_mean(i,:) = new_mean(i,:)/sum(gama(:,i)); 
    end
    new_cov = cell(1,k3); 
    for i= 1 : k3
        tmp_cov = zeros(64,64);
        for z = 1 : size(A,1)
            tmp_cov = tmp_cov + gama(z,i)*(A(z,:) - new_mean(i,:))'*(A(z,:) - new_mean(i,:));
        end
        new_cov{1,i} = tmp_cov/sum(gama(:,i)); 
        new_cov{1,i} = diag(diag(new_cov{1,i}));
    end
    new_W = zeros(k3,1);
    for i= 1 : k3
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    L_new = 0;
    for i= 1: size_train_class3
        p_given_theta = 1;
        for z =  1: size(train3_cell{i,1},1)
            temp = 0;
            for j = 1 : k3
                temp = temp + new_W(j)*exp(-0.5*(train3_cell{i,:}(z,:)-new_mean(j,:))*inv(new_cov{1,j})*(train3_cell{i,:}(z,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{3,1} = mean_value;
theta{3,2} = cov;
theta{3,3} = W;

%train class4--------------------------------------------------------------------------------------
clearvars data data_per_cluster gama term;
A = train_data4;
total_data = size(A,1);

[index,mean_value] = kmeans(A,k4);

data_per_cluster = ones(k4,1);
data = cell(1,k4);

for i=1:total_data
    tmp = data{1,(index(i))};
    tmp(data_per_cluster(index(i)),:) = A(i,:);
    data{1,(index(i))} = tmp;
    data_per_cluster(index(i)) = data_per_cluster(index(i)) + 1;
end

cov = cell(1,k4);
W = zeros(k4,1);
for i=1 : k4
    cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
    cov{1,i} = diag(diag(cov{1,i}));
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1: size_train_class4
        p_given_theta = 1;
        for z =  1: size(train4_cell{i,1},1)
            temp = 0;
            for j = 1 : k4
                temp = temp + W(j)*exp(-0.5*(train4_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train4_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
end

while abs(L_old - L_new)>10
    L_old = L_new;
    gama = zeros(total_data,k4);
    for i= 1 : total_data
        for j= 1: k4
            term(j) = W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        for j= 1: k4
            gama(i,j) = term(j)/sum(term);
        end
    end
    
    new_mean = gama' * A;
    for i= 1 : k4
        new_mean(i,:) = new_mean(i,:)/sum(gama(:,i)); 
    end
    new_cov = cell(1,k4); 
    for i= 1 : k4
        tmp_cov = zeros(64,64);
        for z = 1 : size(A,1)
            tmp_cov = tmp_cov + gama(z,i)*(A(z,:) - new_mean(i,:))'*(A(z,:) - new_mean(i,:));
        end
        new_cov{1,i} = tmp_cov/sum(gama(:,i)); 
        new_cov{1,i} = diag(diag(new_cov{1,i}));
    end
    new_W = zeros(k4,1);
    for i= 1 : k4
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    L_new = 0;
    for i= 1: size_train_class4
        p_given_theta = 1;
        for z =  1: size(train4_cell{i,1},1)
            temp = 0;
            for j = 1 : k4
                temp = temp + new_W(j)*exp(-0.5*(train4_cell{i,:}(z,:)-new_mean(j,:))*inv(new_cov{1,j})*(train4_cell{i,:}(z,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{4,1} = mean_value;
theta{4,2} = cov;
theta{4,3} = W;

%train class5--------------------------------------------------------------------------------------
clearvars data data_per_cluster gama term;
A = train_data5;
total_data = size(A,1);

[index,mean_value] = kmeans(A,k5);

data_per_cluster = ones(k5,1);
data = cell(1,k5);

for i=1:total_data
    tmp = data{1,(index(i))};
    tmp(data_per_cluster(index(i)),:) = A(i,:);
    data{1,(index(i))} = tmp;
    data_per_cluster(index(i)) = data_per_cluster(index(i)) + 1;
end

cov = cell(1,k5);
W = zeros(k5,1);
for i=1 : k5
    cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
    cov{1,i} = diag(diag(cov{1,i}));
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1: size_train_class5
        p_given_theta = 1;
        for z =  1: size(train5_cell{i,1},1)
            temp = 0;
            for j = 1 : k5
                temp = temp + W(j)*exp(-0.5*(train5_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train5_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
end

while abs(L_old - L_new)>10
    L_old = L_new;
    gama = zeros(total_data,k5);
    for i= 1 : total_data
        for j= 1: k5
            term(j) = W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        for j= 1: k5
            gama(i,j) = term(j)/sum(term);
        end
    end
    
    new_mean = gama' * A;
    for i= 1 : k5
        new_mean(i,:) = new_mean(i,:)/sum(gama(:,i)); 
    end
    new_cov = cell(1,k5); 
    for i= 1 : k5
        tmp_cov = zeros(64,64);
        for z = 1 : size(A,1)
            tmp_cov = tmp_cov + gama(z,i)*(A(z,:) - new_mean(i,:))'*(A(z,:) - new_mean(i,:));
        end
        new_cov{1,i} = tmp_cov/sum(gama(:,i)); 
        new_cov{1,i} = diag(diag(new_cov{1,i}));
    end
    new_W = zeros(k5,1);
    for i= 1 : k5
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    L_new = 0;
    for i= 1: size_train_class5
        p_given_theta = 1;
        for z =  1: size(train5_cell{i,1},1)
            temp = 0;
            for j = 1 : k5
                temp = temp + new_W(j)*exp(-0.5*(train5_cell{i,:}(z,:)-new_mean(j,:))*inv(new_cov{1,j})*(train5_cell{i,:}(z,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
            end
            p_given_theta = p_given_theta + log(temp);
        end
        L_new = L_new + p_given_theta;
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{5,1} = mean_value;
theta{5,2} = cov;
theta{5,3} = W;


%predict train-------------------------------------------------------------------------------------------
train_cell = [train1_cell;train2_cell;train3_cell;train4_cell;train5_cell];
size_train_class = [size_train_class1,size_train_class2,size_train_class3,size_train_class4,size_train_class5];
total_data = sum(size_train_class);

Actual_class = [ones(size_train_class1,1);2.*ones(size_train_class2,1);3.*ones(size_train_class3,1);4.*ones(size_train_class4,1);5.*ones(size_train_class5,1)];
predicted_class = zeros(total_data,1);

train_error = 0;

for i= 1 : total_data
    likelihood = zeros(5,1);
    
    mean_value = theta{1,1};
    cov = theta{1,2};
    W = theta{1,3};
    p_given_theta = 1;
    for z =  1: size(train_cell{i,1},1)
        tmp = 0;
        for j = 1 : k1
            tmp = tmp + W(j)*exp(-0.5*(train_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(1) = p_given_theta * (size_train_class1/total_data);

    mean_value = theta{2,1};
    cov = theta{2,2};
    W = theta{2,3};
    p_given_theta = 1;
    for z =  1: size(train_cell{i,1},1)
        tmp = 0;
        for j = 1 : k2
            tmp = tmp + W(j)*exp(-0.5*(train_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(2) = p_given_theta * (size_train_class2/total_data);
    
    mean_value = theta{3,1};
    cov = theta{3,2};
    W = theta{3,3};
    p_given_theta = 1;
    for z =  1: size(train_cell{i,1},1)
        tmp = 0;
        for j = 1 : k3
            tmp = tmp + W(j)*exp(-0.5*(train_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(3) = p_given_theta * (size_train_class3/total_data);
    
    mean_value = theta{4,1};
    cov = theta{4,2};
    W = theta{4,3};
    p_given_theta = 1;
    for z =  1: size(train_cell{i,1},1)
        tmp = 0;
        for j = 1 : k4
            tmp = tmp + W(j)*exp(-0.5*(train_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(4) = p_given_theta * (size_train_class4/total_data);
    
    mean_value = theta{5,1};
    cov = theta{5,2};
    W = theta{5,3};
    p_given_theta = 1;
    for z =  1: size(train_cell{i,1},1)
        tmp = 0;
        for j = 1 : k5
            tmp = tmp + W(j)*exp(-0.5*(train_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(train_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(5) = p_given_theta * (size_train_class5/total_data);
    
    [value,index] = max(likelihood);
    predicted_class(i) = index;
    
    if Actual_class(i) ~= predicted_class(i)
        train_error = train_error+1;
    end
end

%predict val-------------------------------------------------------------------------------------------
validation_cell = [validation1_cell;validation2_cell;validation3_cell;validation4_cell;validation5_cell];
size_validation_class = [size_validation_class1,size_validation_class2,size_validation_class3,size_validation_class4,size_validation_class5];
total_data = sum(size_validation_class);

Actual_class_val = [ones(size_validation_class1,1);2.*ones(size_validation_class2,1);3.*ones(size_validation_class3,1);4.*ones(size_validation_class4,1);5.*ones(size_validation_class5,1)];
predicted_class_val = zeros(total_data,1);

validation_error = 0;

for i= 1 : total_data
    likelihood = zeros(5,1);
    
    mean_value = theta{1,1};
    cov = theta{1,2};
    W = theta{1,3};
    p_given_theta = 1;
    for z =  1: size(validation_cell{i,1},1)
        tmp = 0;
        for j = 1 : k1
            tmp = tmp + W(j)*exp(-0.5*(validation_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(validation_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(1) = p_given_theta * (size_validation_class1/total_data);

    mean_value = theta{2,1};
    cov = theta{2,2};
    W = theta{2,3};
    p_given_theta = 1;
    for z =  1: size(validation_cell{i,1},1)
        tmp = 0;
        for j = 1 : k2
            tmp = tmp + W(j)*exp(-0.5*(validation_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(validation_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(2) = p_given_theta * (size_validation_class2/total_data);
    
    mean_value = theta{3,1};
    cov = theta{3,2};
    W = theta{3,3};
    p_given_theta = 1;
    for z =  1: size(validation_cell{i,1},1)
        tmp = 0;
        for j = 1 : k3
            tmp = tmp + W(j)*exp(-0.5*(validation_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(validation_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(3) = p_given_theta * (size_validation_class3/total_data);
    
    mean_value = theta{4,1};
    cov = theta{4,2};
    W = theta{4,3};
    p_given_theta = 1;
    for z =  1: size(validation_cell{i,1},1)
        tmp = 0;
        for j = 1 : k4
            tmp = tmp + W(j)*exp(-0.5*(validation_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(validation_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(4) = p_given_theta * (size_validation_class4/total_data);
    
    mean_value = theta{5,1};
    cov = theta{5,2};
    W = theta{5,3};
    p_given_theta = 1;
    for z =  1: size(validation_cell{i,1},1)
        tmp = 0;
        for j = 1 : k5
            tmp = tmp + W(j)*exp(-0.5*(validation_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(validation_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(5) = p_given_theta * (size_validation_class5/total_data);
    
    [value,index] = max(likelihood);
    predicted_class_val(i) = index;
    
    if Actual_class_val(i) ~= predicted_class_val(i)
        validation_error = validation_error+1;
    end
end

%predict test-------------------------------------------------------------------------------------------
test_cell = [test1_cell;test2_cell;test3_cell;test4_cell;test5_cell];
size_test_class = [size_test_class1,size_test_class2,size_test_class3,size_test_class4,size_test_class5];
total_data = sum(size_test_class);

Actual_class_test = [ones(size_test_class1,1);2.*ones(size_test_class2,1);3.*ones(size_test_class3,1);4.*ones(size_test_class4,1);5.*ones(size_test_class5,1)];
predicted_class_test = zeros(total_data,1);

test_error = 0;

for i= 1 : total_data
    likelihood = zeros(5,1);
    
    mean_value = theta{1,1};
    cov = theta{1,2};
    W = theta{1,3};
    p_given_theta = 1;
    for z =  1: size(test_cell{i,1},1)
        tmp = 0;
        for j = 1 : k1
            tmp = tmp + W(j)*exp(-0.5*(test_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(test_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(1) = p_given_theta * (size_test_class1/total_data);

    mean_value = theta{2,1};
    cov = theta{2,2};
    W = theta{2,3};
    p_given_theta = 1;
    for z =  1: size(test_cell{i,1},1)
        tmp = 0;
        for j = 1 : k2
            tmp = tmp + W(j)*exp(-0.5*(test_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(test_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(2) = p_given_theta * (size_test_class2/total_data);
    
    mean_value = theta{3,1};
    cov = theta{3,2};
    W = theta{3,3};
    p_given_theta = 1;
    for z =  1: size(test_cell{i,1},1)
        tmp = 0;
        for j = 1 : k3
            tmp = tmp + W(j)*exp(-0.5*(test_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(test_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(3) = p_given_theta * (size_test_class3/total_data);
    
    mean_value = theta{4,1};
    cov = theta{4,2};
    W = theta{4,3};
    p_given_theta = 1;
    for z =  1: size(test_cell{i,1},1)
        tmp = 0;
        for j = 1 : k4
            tmp = tmp + W(j)*exp(-0.5*(test_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(test_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(4) = p_given_theta * (size_test_class4/total_data);
    
    mean_value = theta{5,1};
    cov = theta{5,2};
    W = theta{5,3};
    p_given_theta = 1;
    for z =  1: size(test_cell{i,1},1)
        tmp = 0;
        for j = 1 : k5
            tmp = tmp + W(j)*exp(-0.5*(test_cell{i,:}(z,:)-mean_value(j,:))*inv(cov{1,j})*(test_cell{i,:}(z,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        p_given_theta = p_given_theta * log(tmp);
    end
    likelihood(5) = p_given_theta * (size_test_class5/total_data);
    
    [value,index] = max(likelihood);
    predicted_class_test(i) = index;
    
    if Actual_class_test(i) ~= predicted_class_test(i)
        test_error = test_error+1;
    end
end

acc_train = (sum(size_train_class) - train_error)*100/sum(size_train_class);
acc_val = (sum(size_validation_class) - validation_error)*100/sum(size_validation_class);
acc_test = (sum(size_test_class) - test_error)*100/sum(size_test_class);

%confusion matrix----------------------------------------------------------
confusion_mat_train = zeros(5,5);
count = 0;

predicted_class1 = predicted_class(count+1:count+size_train_class1);
count = count + size_train_class1;
predicted_class2 = predicted_class(count+1:count+size_train_class2);
count = count + size_train_class2;
predicted_class3 = predicted_class(count+1:count+size_train_class3);
count = count + size_train_class3;
predicted_class4 = predicted_class(count+1:count+size_train_class4);
count = count + size_train_class4;
predicted_class5 = predicted_class(count+1:count+size_train_class5);

for i=1:5
    confusion_mat_train(1,i) = size(predicted_class1(predicted_class1==i),1);
end
for i=1:5
    confusion_mat_train(2,i) = size(predicted_class2(predicted_class2==i),1);
end
for i=1:5
    confusion_mat_train(3,i) = size(predicted_class3(predicted_class3==i),1);
end
for i=1:5
    confusion_mat_train(4,i) = size(predicted_class4(predicted_class4==i),1);
end
for i=1:5
    confusion_mat_train(5,i) = size(predicted_class5(predicted_class5==i),1);
end

confusion_mat_test = zeros(5,5);
count = 0;

predicted_class_test1 = predicted_class_test(count+1:count+size_test_class1);
count = count + size_test_class1;
predicted_class_test2 = predicted_class_test(count+1:count+size_test_class2);
count = count + size_test_class2;
predicted_class_test3 = predicted_class_test(count+1:count+size_test_class3);
count = count + size_test_class3;
predicted_class_test4 = predicted_class_test(count+1:count+size_test_class4);
count = count + size_test_class4;
predicted_class_test5 = predicted_class_test(count+1:count+size_test_class5);

for i=1:5
    confusion_mat_test(1,i) = size(predicted_class_test1(predicted_class_test1==i),1);
end
for i=1:5
    confusion_mat_test(2,i) = size(predicted_class_test2(predicted_class_test2==i),1);
end
for i=1:5
    confusion_mat_test(3,i) = size(predicted_class_test3(predicted_class_test3==i),1);
end
for i=1:5
    confusion_mat_test(4,i) = size(predicted_class_test4(predicted_class_test4==i),1);
end
for i=1:5
    confusion_mat_test(5,i) = size(predicted_class_test5(predicted_class_test5==i),1);
end
