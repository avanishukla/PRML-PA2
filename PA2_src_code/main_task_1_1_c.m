clc;
clearvars;
mydir = 'D:\prml\datasets 1_2\datasets 1_2\group8\overlapping\';
fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x1 = A(1:2:end);
y1 = A(2:2:end);
D = ones(length(x1),1);
X1 = [x1 y1 D];
fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x2 = A(1:2:end);
y2 = A(2:2:end);
D = 2.*ones(length(x2),1);
X2 = [x2 y2 D];


fileID = fopen(strcat(mydir,'class3_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x3 = A(1:2:end);
y3 = A(2:2:end);
D = 3.*ones(length(x3),1);
X3 = [x3 y3 D];

c = vertcat(X1,X2,X3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 15;  %%%%%%%%%best case  - k = 15
%12 20 25 27 30 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure(4);

gscatter(c(:,1),c(:,2),c(:,3),'rgb','',19)
xlabel('x-value')
ylabel('y-value')
str  = { strcat('Decision Region Plot with k = ' ,num2str(k))};
title(str,'FontSize',15);
%[15 15],[15 -15]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
calculated_label_train = zeros(length(3*x1),1);
%error calculation for train data for class 1 %
confusion_mat_train =  zeros(3,3);
dist_matrix = zeros(750,2);
index = 1;
for i = 1 :length(x1)
    for j = 1:750
        dist_matrix(j,1) =  distance(x1(i,1),y1(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
        calculated_label_train(index,1) = 1;
    elseif label_count == c2
        confusion_mat_train(1,2) = confusion_mat_train(1,2)+1;
        calculated_label_train(index,1) = 2;
    else
        confusion_mat_train(1,3) = confusion_mat_train(1,3)+1;
        calculated_label_train(index,1) = 3;
    end
    index = index + 1;
end
    
%error calculation for train data for class 2 %

dist_matrix = zeros(750,2);
for i = 1 :length(x2)
    for j = 1:750
        dist_matrix(j,1) =  distance(x2(i,1),y2(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
        calculated_label_train(index,1) = 1;
    elseif label_count == c2
        confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
        calculated_label_train(index,1) = 2;
    else
        confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
        calculated_label_train(index,1) = 3;
    end
    index = index + 1;
end

%error calculation for train data for class 3 %

dist_matrix = zeros(750,2);
for i = 1 :length(x3)
    for j = 1:750
        dist_matrix(j,1) =  distance(x3(i,1),y3(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
        %disp(c1)
        %disp(c2)
        %disp(c3)
    end
    
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_train(3,1) = confusion_mat_train(3,1)+1;
        calculated_label_train(index,1) = 1;
    elseif label_count == c2
        confusion_mat_train(3,2) = confusion_mat_train(3,2)+1;
        calculated_label_train(index,1) = 2;
    else
        confusion_mat_train(3,3) = confusion_mat_train(3,3)+1;
        calculated_label_train(index,1) = 3;
    end
    index = index + 1;
end


class1_error_train = error(calculated_label_train(1:250),1);
class2_error_train = error(calculated_label_train(251:500),2);
class3_error_train = error(calculated_label_train(501:750),3);
class1_error_percent_train = error_percent(class1_error_train,250);
class2_error_percent_train = error_percent(class2_error_train,250);
class3_error_percent_train = error_percent(class3_error_train,250);

total_error_train =  class1_error_train + class2_error_train + class3_error_train;
total_error_train_per = error_percent(total_error_train,750);
accuracy_train = cal_accuracy(confusion_mat_train,750);
%%%%%%%%%%%%%%%to show object in wrong class%%%%%%%%%%%%
%{
D = ones(length(x1),1);
D1 = 2*ones(length(x1),1);
D2 = 3*ones(length(x1),1);
train1 = [x1, y1,D]; 
train2 = [x2, y2,D1]; 
train3 = [x3, y3,D2];
train = vertcat(train1,train2,train3);

figure(1);
gscatter(c(:,1),c(:,2),c(:,3),'myk','',24)
xlabel('x-value')
ylabel('y-value')
hold on;
gscatter(train(:,1),train(:,2),train(:,3),'rgb','',9)
xlabel('x-value')
ylabel('y-value')

%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%label cal for validate data

fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x1_val = A(1:2:end);
y1_val = A(2:2:end);
dist_matrix = zeros(750,2);
index = 1;
calculated_label_val = zeros(length(3*x1_val),1);
confusion_mat_val =  zeros(3,3);
%label calculation for class 1 val data
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x1_val(i,1),y1_val(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_val(1,1) = confusion_mat_val(1,1)+1;
        calculated_label_val(index,1) = 1;
    elseif label_count == c2
        confusion_mat_val(1,2) = confusion_mat_val(1,2)+1;
        calculated_label_val(index,1) = 2;
    else
        confusion_mat_val(1,3) = confusion_mat_val(1,3)+1;
        calculated_label_val(index,1) = 3;
    end
    index = index + 1;
end
%label calculation for class 2 val data
fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x2_val = A(1:2:end);
y2_val = A(2:2:end);
dist_matrix = zeros(750,2);
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x2_val(i,1),y2_val(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_val(2,1) = confusion_mat_val(1,1)+1;
        calculated_label_val(index,1) = 1;
    elseif label_count == c2
        confusion_mat_val(2,2) = confusion_mat_val(2,2)+1;
        calculated_label_val(index,1) = 2;
    else
        confusion_mat_val(2,3) = confusion_mat_val(2,3)+1;
        calculated_label_val(index,1) = 3;
    end
    index = index + 1;
end

%label calculation for class 3 val data
fileID = fopen(strcat(mydir,'class3_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x3_val = A(1:2:end);
y3_val = A(2:2:end);
dist_matrix = zeros(750,2);
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x3_val(i,1),y3_val(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        confusion_mat_val(3,1) = confusion_mat_val(3,1)+1;
        calculated_label_val(index,1) = 1;
    elseif label_count == c2
        confusion_mat_val(3,2) = confusion_mat_val(3,2)+1;
        calculated_label_val(index,1) = 2;
    else
        confusion_mat_val(3,3) = confusion_mat_val(3,3)+1;
        calculated_label_val(index,1) = 3;
    end
    index = index + 1;
end


class1_error_val = error(calculated_label_val(1:100),1);
class2_error_val = error(calculated_label_val(101:200),2);
class3_error_val = error(calculated_label_val(201:300),3);
class1_error_percent_val = error_percent(class1_error_val,100);
class2_error_percent_val = error_percent(class2_error_val,100);
class3_error_percent_val = error_percent(class3_error_val,100);

total_error_val =  class1_error_val + class2_error_val + class3_error_val;
total_error_val_per = error_percent(total_error_val,300);
accuracy_val = cal_accuracy(confusion_mat_val,300);

fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x1_test = A(1:2:end);
y1_test = A(2:2:end);
dist_matrix = zeros(750,2);
index = 1;
calculated_label_test = zeros(length(3*x1_test),1);
confusion_mat_test =  zeros(3,3);
%label calculation for class 1 test data
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x1_test(i,1),y1_test(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
  
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
    if label_count == c1
        calculated_label_test (index,1) = 1;
        confusion_mat_test(1,1) = confusion_mat_test(1,1)+1;
    elseif label_count == c2
       calculated_label_test(index,1) = 2;
       confusion_mat_test(1,2) = confusion_mat_test(1,2)+1;
    else
       calculated_label_test(index,1) = 3;
       confusion_mat_test(1,3) = confusion_mat_test(1,3)+1;
    end
    index = index + 1;
end
%label calculation for class 2 test data
fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x2_test = A(1:2:end);
y2_test = A(2:2:end);
dist_matrix = zeros(750,2);
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x2_test(i,1),y2_test(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
     if label_count == c1
       calculated_label_test(index,1) = 1;
       confusion_mat_test(2,1) = confusion_mat_test(2,1)+1;
    elseif label_count == c2
        calculated_label_test(index,1) = 2;
        confusion_mat_test(2,2) = confusion_mat_test(2,2)+1;
    else
        calculated_label_test(index,1) = 3;
        confusion_mat_test(2,3) = confusion_mat_test(2,3)+1;
    end
    index = index + 1;
end

%label calculation for class 3 test data
fileID = fopen(strcat(mydir,'class3_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x3_test = A(1:2:end);
y3_test = A(2:2:end);
dist_matrix = zeros(750,2);
for i = 1:100
    for j = 1:750
        dist_matrix(j,1) =  distance(x3_test(i,1),y3_test(i,1),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        else
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
     if label_count == c1
       calculated_label_test(index,1) = 1;
       confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == c2
        calculated_label_test(index,1) = 2;
        confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
    else
        calculated_label_test(index,1) = 3;
        confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
    index = index + 1;
end




test_data1 = [x1_test, y1_test,calculated_label_test(1:100)]; 
test_data2 = [x2_test, y2_test,calculated_label_test(101:200)]; 
test_data3 = [x3_test, y3_test,calculated_label_test(201:300)]; 
test_data = vertcat(test_data1,test_data2,test_data3);
D = ones(length(x1_test),1);
D1 = 2*ones(length(x1_test),1);
D2 = 3*ones(length(x1_test),1);
test1 = [x1_test, y1_test,D]; 
test2 = [x2_test, y2_test,D1]; 
test3 = [x3_test, y3_test,D2];
test = vertcat(test1,test2,test3);

%%%%%%%%%%%%%%%to show object in wrong class%%%%%%%%%%%%
%{
gscatter(test_data(:,1),test_data(:,2),test_data(:,3),'myk','',34)
xlabel('x-value')
ylabel('y-value')
hold on;
gscatter(test(:,1),test(:,2),test(:,3),'rgb','',12)
xlabel('x-value')
ylabel('y-value')
%}
class1_error_test = error(calculated_label_test(1:100),1);
class2_error_test = error(calculated_label_test(101:200),2);
class3_error_test = error(calculated_label_test(201:300),3);
class1_error_percent_test = error_percent(class1_error_test,100);
class2_error_percent_test = error_percent(class2_error_test,100);
class3_error_percent_test = error_percent(class3_error_test,100);
total_error_test =  class1_error_test + class2_error_test + class3_error_test;
total_error_test_per = error_percent(total_error_test,300);
%disp(confusion_mat_test);
accuracy_test = cal_accuracy(confusion_mat_test,300);

%plot---------------------------------------------------------------------------------------------

xaxis = -10:0.05:15;
yaxis = -15:0.05:15;
[x, y] = meshgrid(xaxis, yaxis);
image_size = size(x);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);
index = 1;
for i= 1: total_plot_data
    
    for j = 1:750
        dist_matrix(j,1) =  distance(plot_data(i,1),plot_data(i,2),c(j,1),c(j,2)) ;
        dist_matrix(j,2) = c(j,3);
    end
    sortedmat = sortrows(dist_matrix, 1);
    c1 =0; 
    c2 =0;
    c3 = 0;
    for l = 1:k
        if sortedmat(l,2) == 1
            c1 = c1 + 1 ;
        elseif  sortedmat(l,2) == 2
             c2 = c2 +1 ;
        elseif sortedmat(l,2) == 3
             c3 = c3 + 1;
        end
    end
    label_count = max(max(c1,c2),c3);
     if label_count == c1
       predicted_plot(index,1) = 1;
       %confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == c2
        predicted_plot(index,1) = 2;
       % confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
     else
        predicted_plot(index,1) = 3;
        %confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
    index = index + 1;
end

plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');
hold on
scatter(X1(:,1),X1(:,2),'k','x','LineWidth',3);
scatter(X2(:,1),X2(:,2),'r','x','LineWidth',3);
scatter(X3(:,1),X3(:,2),'b','x','LineWidth',3);
xlabel('x1','FontSize',15)
ylabel('x2','FontSize',15)
str  = { strcat('Decision Region Plot for overlapping data with k = ' ,num2str(k))};
title(str,'FontSize',15);
t = text(-5,-10,'CLASS 1','Color','r','FontSize',14);
t1 = text(3,10,'CLASS 2','Color','r','FontSize',14);
t3 = text(3,-10,'CLASS 3','Color','b','FontSize',14);
hold off;

function euclideanDistance = distance(x1, y1, x2, y2) 
euclideanDistance = sqrt((x2-x1)^2+(y2-y1)^2);
end


function error_count = error(x,label)
    error_count = 0;
    for i= 1:size(x)
        if x(i,1) ~= label
            %disp(i);
            error_count = error_count + 1;
        end
    end
end

function error_p = error_percent(error,total)
    error_p = (error/total)*100;
end

function accuracy = cal_accuracy(confusion_mat,total)
    accuracy = (confusion_mat(1,1)+confusion_mat(2,2)+confusion_mat(3,3))/total;
end
