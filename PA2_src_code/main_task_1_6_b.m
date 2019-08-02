clc;
clearvars;
mydir = 'D:\prml\datasets 1_2\datasets 1_2\group8\nonlinearly_separable';

fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
val_data2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
val_data2 = val_data2';
D = ones(size(val_data2,1),1);
train_data1 = [val_data2,D];

fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
val_data2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
val_data2 = val_data2';
D = 2.*ones(size(val_data2,1),1);
train_data2 = [val_data2,D];

k =45; %%%%%%%%%%%%% CHECK for accuracy %%%%%%%%%%%% 45 ke upar bigad raha hai
% 10 5 4 3 = 1 1 1
fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
val_data1 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
val_data1 = val_data1';
predicted_val1 = zeros(size(val_data1,1),1);

for i = 1:size(val_data1,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = sqrt((train_data1(j,1)-val_data1(i,1))^2+(train_data1(j,2)-val_data1(i,2))^2);
        dist_matrix(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = sqrt((train_data2(j,1)-val_data1(i,1))^2+(train_data2(j,2)-val_data1(i,2))^2);
        dist_matrix(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_val1(i) = dist(index,2);
end

fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
val_data2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
val_data2 = val_data2';
predicted_val2 = zeros(size(val_data2,1),1);

for i = 1:size(val_data2,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = sqrt((train_data1(j,1)-val_data2(i,1))^2+(train_data1(j,2)-val_data2(i,2))^2);
        dist_matrix(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = sqrt((train_data2(j,1)-val_data2(i,1))^2+(train_data2(j,2)-val_data2(i,2))^2);
        dist_matrix(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_val2(i) = dist(index,2);
end
%-----------------------------------------------------------------------------------------
fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
test_data1 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
test_data1 = test_data1';
predicted_test1 = zeros(size(test_data1,1),1);

for i = 1:size(test_data1,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = sqrt((train_data1(j,1)-test_data1(i,1))^2+(train_data1(j,2)-test_data1(i,2))^2);
        dist_matrix(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = sqrt((train_data2(j,1)-test_data1(i,1))^2+(train_data2(j,2)-test_data1(i,2))^2);
        dist_matrix(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_test1(i) = dist(index,2);
end

fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
test_data2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
test_data2 = test_data2';
predicted_test2 = zeros(size(test_data2,1),1);

for i = 1:size(test_data2,1)
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = sqrt((train_data1(j,1)-test_data2(i,1))^2+(train_data1(j,2)-test_data2(i,2))^2);
        dist_matrix(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = sqrt((train_data2(j,1)-test_data2(i,1))^2+(train_data2(j,2)-test_data2(i,2))^2);
        dist_matrix(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_test2(i) = dist(index,2);
end
%%%%%%%%%%%%%%%%%%%%%%%% added by rt


dist_matrix_train = zeros(250,2);
dist_train1 = zeros(2,2) ;

predicted_train1 = zeros(size(train_data1,1),1);

for i = 1:size(train_data1,1)
    dist_train1 = zeros(2,2);
    dist_matrix_train = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix_train(j,1) = sqrt((train_data1(j,1)-train_data1(i,1))^2+(train_data1(j,2)-train_data1(i,2))^2);
        dist_matrix_train(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix_train, 1);
    dist_train1(1,:) = sortedmat(k,:);
    dist_matrix_train = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix_train(j,1) = sqrt((train_data2(j,1)-train_data1(i,1))^2+(train_data2(j,2)-train_data1(i,2))^2);
        dist_matrix_train(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix_train, 1);
    dist_train1(2,:) = sortedmat(k,:);
    [value,index] = min(dist_train1(:,1));
    predicted_train1(i) = dist_train1(index,2);
end

predicted_train2 = zeros(size(train_data2,1),1);

for i = 1:size(train_data2,1)
    dist_train = zeros(2,2);
    dist_matrix_train = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix_train(j,1) = sqrt((train_data1(j,1)-train_data2(i,1))^2+(train_data1(j,2)-train_data2(i,2))^2);
        dist_matrix_train(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix_train, 1);
    dist_train(1,:) = sortedmat(k,:);
    dist_matrix_train = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix_train(j,1) = sqrt((train_data2(j,1)-train_data2(i,1))^2+(train_data2(j,2)-train_data2(i,2))^2);
        dist_matrix_train(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix_train, 1);
    dist_train(2,:) = sortedmat(k,:);
    [value,index] = min(dist_train(:,1));
    predicted_train2(i) = dist_train(index,2);
end

%%%%%%-------------confusion matrices
count = 0;
confusion_mat_test =  zeros(2,2);
for i = 1 :100
        if(predicted_test1(i,1) == 1)
            confusion_mat_test(1,1) = confusion_mat_test(1,1)+1 ;
        elseif(predicted_test1(i,1) == 2)
            confusion_mat_test(1,2) = confusion_mat_test(1,2)+1 ;
        end
    
        if(predicted_test2(i,1) == 1)
            confusion_mat_test(2,1) = confusion_mat_test(2,1)+1 ;
        elseif(predicted_test2(i,1) == 2)
            confusion_mat_test(2,2) = confusion_mat_test(2,2)+1 ;
        end    
end
confusion_mat_val =  zeros(2,2);
for i = 1 :150
        if(predicted_val1(i,1) == 1)
            confusion_mat_val(1,1) = confusion_mat_val(1,1)+1 ;
        elseif(predicted_val1(i,1) == 2)
            confusion_mat_val(1,2) = confusion_mat_val(1,2)+1 ;
        end
    
        if(predicted_val2(i,1) == 1)
            confusion_mat_val(2,1) = confusion_mat_val(2,1)+1 ;
        elseif(predicted_val2(i,1) == 2)
            confusion_mat_val(2,2) = confusion_mat_val(2,2)+1 ;
        end  
end

confusion_mat_train =  zeros(2,2);
for i = 1 :250
        if(predicted_train1(i,1) == 1)
            confusion_mat_train(1,1) = confusion_mat_train(1,1)+1 ;
        elseif(predicted_train1(i,1) == 2)
            confusion_mat_train(1,2) = confusion_mat_train(1,2)+1 ;
        end
    
        if(predicted_train2(i,1) == 1)
            confusion_mat_train(2,1) = confusion_mat_train(2,1)+1 ;
        elseif(predicted_train2(i,1) == 2)
            confusion_mat_train(2,2) = confusion_mat_train(2,2)+1 ;
        end  
end

%%%%%%%%%%%%%%%%%

%-----------------------------------------------------------------------------------------
test_data = [test_data1;test_data2];
predicted_class = [predicted_test1;predicted_test2];
end1 =1;end2=1;
for i = 1: size(predicted_class)
   if predicted_class(i)==1
       predicted_class1(end1,:) = test_data(i,:);
       end1 = end1+1;
   elseif predicted_class(i)==2
       predicted_class2(end2,:) = test_data(i,:);
       end2 = end2+1;
   end 
end

error1 = error(predicted_test1,1);
error2 = error(predicted_test2,2);

accuracy = (confusion_mat_test(1,1)+  confusion_mat_test(2,2)) / 200 ;
%{
%%%%%%%%%5 plot
xaxis = -1.7:0.005:2.7;
yaxis = -1.7:0.005:1.5;

[x, y] = meshgrid(xaxis, yaxis);
image_size = size(x);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);

for i= 1: total_plot_data
    i
    dist = zeros(2,2);
    dist_matrix = zeros(size(train_data1,1),2);
    for j = 1:size(train_data1,1)
        dist_matrix(j,1) = sqrt((train_data1(j,1)-plot_data(i,1))^2+(train_data1(j,2)-plot_data(i,2))^2);
        dist_matrix(j,2) = train_data1(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(1,:) = sortedmat(k,:);
    dist_matrix = zeros(size(train_data2,1),2);
    for j = 1:size(train_data2,1)
        dist_matrix(j,1) = sqrt((train_data2(j,1)-plot_data(i,1))^2+(train_data2(j,2)-plot_data(i,2))^2);
        dist_matrix(j,2) = train_data2(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    dist(2,:) = sortedmat(k,:);
   % dist_matrix = zeros(size(train_data3,1),2);
    %for j = 1:size(train_data3,1)
     %   dist_matrix(j,1) = sqrt((train_data3(j,1)-plot_data(i,1))^2+(train_data3(j,2)-plot_data(i,2))^2);
    %    dist_matrix(j,2) = train_data3(j,3);
   % end
    sortedmat = sortrows(dist_matrix, 1);
    %dist(3,:) = sortedmat(k,:);
    [value,index] = min(dist(:,1));
    predicted_plot(i) = dist(index,2);
end

plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');
hold on

scatter(train_data1(:,1),train_data1(:,2),'m','x');
%hold on;
scatter(train_data2(:,1),train_data2(:,2),'w','x');
%scatter(train_data3(:,1),train_data3(:,2),'r','x');
hold off;
%%%%%%%%%%%%%%%%%%%%
%{

scatter(test_data1(:,1),test_data1(:,2),'m','.');
hold on;
scatter(test_data2(:,1),test_data2(:,2),'g','.');
scatter(test_data3(:,1),test_data3(:,2),'r','.');
scatter(predicted_class1(:,1),predicted_class1(:,2),'m');
scatter(predicted_class2(:,1),predicted_class2(:,2),'g');
scatter(predicted_class3(:,1),predicted_class3(:,2),'r');
hold off;
%}
%}
accuracy_train =( confusion_mat_train(1,1)+confusion_mat_train(2,2))/500  ;
accuracy_test = (confusion_mat_test(1,1)+confusion_mat_test(2,2))/200  ;
accuracy_valid= (confusion_mat_val(1,1)+confusion_mat_val(2,2))/300 ;

function error_count = error(x,label)
    error_count = 0;
    for i= 1:size(x)
        if x(i,1) ~= label
            %disp(i);
            error_count = error_count + 1;
        end
    end
end
