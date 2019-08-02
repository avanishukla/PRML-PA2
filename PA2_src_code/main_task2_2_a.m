clc;
clearvars;
mydir = 'D:\prml\datasets 1_2\datasets 1_2\group8\2a\';
fileID = fopen(strcat(mydir,'dataset.txt'),'r');
A = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f','Delimiter',',');

all_data = zeros(size(A{1,1},1),11);

for i=1:11
       all_data(:,i) = A{1,i}; 
end 
fclose(fileID);

size_train_data = floor(0.7*size(all_data,1));
size_validation_data = floor(0.1*size(all_data,1));
size_test_data = size(all_data,1) - size_train_data - size_validation_data;
train_data = all_data(1:size_train_data,:);
validation_data = all_data(size_train_data+1:size_train_data+size_validation_data,:);
test_data = all_data(size_train_data+size_validation_data+1:size_train_data+size_validation_data+size_test_data,:);

train1_data = zeros(size(train_data(train_data(:,11)==2),1),10);
train2_data = zeros(size(train_data(train_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(train_data)
   if train_data(i,11)==2
       train1_data(end1,:) = train_data(i,1:10);
       end1 = end1+1;
   elseif train_data(i,11)==4
       train2_data(end2,:) = train_data(i,1:10);
       end2 = end2+1;
   end 
end

validation1_data = zeros(size(validation_data(validation_data(:,11)==2),1),10);
validation2_data = zeros(size(validation_data(validation_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(validation_data)
   if validation_data(i,11)==2
       validation1_data(end1,:) = validation_data(i,1:10);
       end1 = end1+1;
   elseif validation_data(i,11)==4
       validation2_data(end2,:) = validation_data(i,1:10);
       end2 = end2+1;
   end 
end

test1_data = zeros(size(test_data(test_data(:,11)==2),1),10);
test2_data = zeros(size(test_data(test_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(test_data)
   if test_data(i,11)==2
       test1_data(end1,:) = test_data(i,1:10);
       end1 = end1+1;
   elseif test_data(i,11)==4
       test2_data(end2,:) = test_data(i,1:10);
       end2 = end2+1;
   end 
end

theta = cell(2,3);
k1 = 4;
k2 = 3;
%train class1--------------------------------------------------------------
A = train1_data;
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
    W(i) = (data_per_cluster(i)-1)/total_data;
end

L_old = Inf;
L_new = 0;
for i= 1 : total_data
    tmp = 0;
   for j = 1 : k1
       tmp = tmp + W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
   end
   L_new = L_new + log(tmp);
end

while abs(L_old - L_new)>0.0000005
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
    %add = [0.0001;0.0001;0.0001;0.0001;0.0001;0.0001;0.0001;0.0001;0.0001;0.0001];
    for i= 1 : k1
        new_cov{1,i} = (A - new_mean(i,:))'*diag(gama(:,i))*(A - new_mean(i,:))/sum(gama(:,i)); 
        new_cov{1,i} = new_cov{1,i}+diag(0.0001.*ones(10,1));
    end
    new_W = zeros(k1,1);
    for i= 1 : k1
        new_W(i) = sum(gama(:,i))/total_data; 
    end
    
    L_new = 0;
    for i= 1 : total_data
        tmp = 0;
        for j = 1 : k1
            tmp = tmp + new_W(j)*exp(-0.5*(A(i,:)-new_mean(j,:))*inv(new_cov{1,j})*(A(i,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
        end
        L_new = L_new + log(tmp);
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{1,1} = mean_value;
theta{1,2} = cov;
theta{1,3} = W;

%train class2-----------------------------------------------------------------------------------------
clearvars A data data_per_cluster gama term;
A = train2_data;
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
    W(i) = (data_per_cluster(i)-1)/total_data;
end
L_old = Inf;
L_new = 0;
for i= 1 : total_data
    tmp = 0;
   for j = 1 : k2
       tmp = tmp + W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
   end
   L_new = L_new + log(tmp);
end

while abs(L_old - L_new)>0.0000005
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
        new_cov{1,i} = (A - new_mean(i,:))'*diag(gama(:,i))*(A - new_mean(i,:))/sum(gama(:,i)); 
        new_cov{1,i} = new_cov{1,i}+diag(0.0001.*ones(10,1));
    end
    new_W = zeros(k2,1);
    for i= 1 : k2
        new_W(i) = sum(gama(:,i))/total_data; 
    end

    L_new = 0;
    for i= 1 : total_data
        tmp = 0;
        for j = 1 : k2
            tmp = tmp + new_W(j)*exp(-0.5*(A(i,:)-new_mean(j,:))*inv(new_cov{1,j})*(A(i,:)-new_mean(j,:))')/(2*pi*sqrt(det(new_cov{1,j})));
        end
        L_new = L_new + log(tmp);
    end
    mean_value = new_mean;
    cov = new_cov;
    W = new_W;
end
disp(L_new);
theta{2,1} = mean_value;
theta{2,2} = cov;
theta{2,3} = W;

k = [k1,k2];

%predict train-------------------------------------------------------------

total_train_data = size(train_data,1);
Actual_class = train_data(:,11);
predicted_class = zeros(total_data,1);
train_dataset = train_data(:,1:10);
train_error = 0;

for i= 1: total_train_data
    max_L = -Inf;
   for p = 1 : 2
        tmp = 0;
        for j = 1 : k(p)
            mean_value = theta{p,1}(j,:);
            cov = theta{p,2}{1,j};
            W = theta{p,3}(j);
            tmp = tmp + W*exp(-0.5*(train_dataset(i,:)-mean_value)*inv(cov)*(train_dataset(i,:)-mean_value)')/(2*pi*sqrt(det(cov)));
        end
        if tmp > max_L
            max_L = tmp;
            predicted_class(i) = 2*p;
        end   
   end
   if Actual_class(i) ~= predicted_class(i)
            train_error = train_error+1;
    end
end
acc_train = (total_train_data - train_error)*100/total_train_data;
%predict validation-----------------------------------------------------------------------------------------------
validation = validation_data(:,1:10);
total_val_data = size(validation,1);
Actual_class_val = validation_data(:,11);
predicted_class_val = zeros(total_val_data,1);
val_error = 0;

for i= 1: total_val_data
    max_L = -Inf;
   for p = 1 : 2
        tmp = 0;
        for j = 1 : k(p)
            mean_value = theta{p,1}(j,:);
            cov = theta{p,2}{1,j};
            W = theta{p,3}(j);
            tmp = tmp + W*exp(-0.5*(validation(i,:)-mean_value)*inv(cov)*(validation(i,:)-mean_value)')/(2*pi*sqrt(det(cov)));
        end
        if tmp > max_L
            max_L = tmp;
            predicted_class_val(i) = 2*p;
        end
   end
   if Actual_class_val(i) ~= predicted_class_val(i)
            val_error = val_error+1;
   end
end
acc_val = (total_val_data - val_error)*100/total_val_data;
%predict test-----------------------------------------------------------------------------------------------
test = test_data(:,1:10);
total_test_data = size(test,1);
Actual_class_test = test_data(:,11);
predicted_class_test = zeros(total_test_data,1);
test_error = 0;

for i= 1: total_test_data
    max_L = -Inf;
   for p = 1 : 2
        tmp = 0;
        for j = 1 : k(p)
            mean_value = theta{p,1}(j,:);
            cov = theta{p,2}{1,j};
            W = theta{p,3}(j);
            tmp = tmp + W*exp(-0.5*(test(i,:)-mean_value)*inv(cov)*(test(i,:)-mean_value)')/(2*pi*sqrt(det(cov)));
        end
        if tmp > max_L
            max_L = tmp;
            predicted_class_test(i) = 2*p;
        end
   end
   if Actual_class_test(i) ~= predicted_class_test(i)
            test_error = test_error+1;
    end
end
acc_test = (total_test_data - test_error)*100/total_test_data;


%confusion matrix----------------------------------------------------------
confusion_mat_train = zeros(2,2);

for i=1:size(predicted_class,1)
    row = Actual_class(i)/2;
    col = predicted_class(i)/2;
    confusion_mat_train(row,col) = confusion_mat_train(row,col) +1; 
end

confusion_mat_test = zeros(2,2);
for i=1:size(predicted_class_test,1)
    row = Actual_class_test(i)/2;
    col = predicted_class_test(i)/2;
    confusion_mat_test(row,col) = confusion_mat_test(row,col) +1; 
end
