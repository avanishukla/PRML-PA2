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

train_data1 = zeros(size(train_data(train_data(:,11)==2),1),10);
train_data2 = zeros(size(train_data(train_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(train_data)
   if train_data(i,11)==2
       train_data1(end1,:) = train_data(i,1:10);
       end1 = end1+1;
   elseif train_data(i,11)==4
       train_data2(end2,:) = train_data(i,1:10);
       end2 = end2+1;
   end 
end

val_data1 = zeros(size(validation_data(validation_data(:,11)==2),1),10);
val_data2 = zeros(size(validation_data(validation_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(validation_data)
   if validation_data(i,11)==2
       val_data1(end1,:) = validation_data(i,1:10);
       end1 = end1+1;
   elseif validation_data(i,11)==4
       val_data2(end2,:) = validation_data(i,1:10);
       end2 = end2+1;
   end 
end

test_data1 = zeros(size(test_data(test_data(:,11)==2),1),10);
test_data2 = zeros(size(test_data(test_data(:,11)==4),1),10);
end1=1;end2=1;
for i = 1: size(test_data)
   if test_data(i,11)==2
       test_data1(end1,:) = test_data(i,1:10);
       end1 = end1+1;
   elseif test_data(i,11)==4
       test_data2(end2,:) = test_data(i,1:10);
       end2 = end2+1;
   end 
end

k = 24;
%train--------------------------------------------------------------------------------
predicted_train1 = zeros(size(train_data1,1),1);

for i = 1:size(train_data1,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-train_data1(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-train_data1(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_train1(i) = dist(index,2);
end

predicted_train2 = zeros(size(train_data2,1),1);

for i = 1:size(train_data2,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-train_data2(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-train_data2(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_train2(i) = dist(index,2);
end
error1 = error(predicted_train1,2);
error2 = error(predicted_train2,4);
train_error = error1+error2;

confusion_mat_train = zeros(2,2);
count = 0;

for i=1:2
    confusion_mat_train(1,i) = size(predicted_train1(predicted_train1==i*2),1);
end
for i=1:2
    confusion_mat_train(2,i) = size(predicted_train2(predicted_train2==i*2),1);
end

%validation-------------------------------------------------------------------------------------------------
predicted_val1 = zeros(size(val_data1,1),1);

for i = 1:size(val_data1,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-val_data1(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-val_data1(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_val1(i) = dist(index,2);
end

predicted_val2 = zeros(size(val_data2,1),1);

for i = 1:size(val_data2,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-val_data2(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-val_data2(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_val2(i) = dist(index,2);
end
error1 = error(predicted_val1,2);
error2 = error(predicted_val2,4);
val_error = error1+error2;

%test-------------------------------------------------------------------------------------------
predicted_test1 = zeros(size(test_data1,1),1);

for i = 1:size(test_data1,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-test_data1(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-test_data1(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_test1(i) = dist(index,2);
end

predicted_test2 = zeros(size(test_data2,1),1);

for i = 1:size(test_data2,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data1(j,z)-test_data2(i,z))^2;
        end
        dist_matrix(j,2) = 2;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = 0;
        for z=1:10
            dist_matrix(j,1) = dist_matrix(j,1) + (train_data2(j,z)-test_data2(i,z))^2;
        end
        dist_matrix(j,2) = 4;
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_test2(i) = dist(index,2);
end
error1 = error(predicted_test1,2);
error2 = error(predicted_test2,4);
test_error = error1+error2;

acc_train = (size_train_data - train_error)*100/size_train_data;
acc_val = (size_validation_data - val_error)*100/size_validation_data;
acc_test = (size_test_data - test_error)*100/size_test_data;



confusion_mat_test = zeros(2,2);

for i=1:2
    confusion_mat_test(1,i) = size(predicted_test1(predicted_test1==i*2),1);
end
for i=1:2
    confusion_mat_test(2,i) = size(predicted_test2(predicted_test2==i*2),1);
end


function error_count = error(x,label)
    error_count = 0;
    for i= 1:size(x)
        if x(i,1) ~= label
            error_count = error_count + 1;
        end
    end
end