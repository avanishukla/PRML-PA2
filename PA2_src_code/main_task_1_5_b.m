clc;
clearvars;
mydir = 'D:\prml\datasets 1_2\datasets 1_2\group8\nonlinearly_separable\';
theta = cell(2,3);

%train class1--------------------------------------------------------------
fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
A = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
A = A';
train1_data = A;
total_data = size(A,1);
k1 = 2;
k2 = 2;
% 3 4 = 1 .9580
%4 3 = 1 .9620
% 3 2 = 1 .9080
% 3  3 = 1 .9180 = 4 3
% 4 4 = 1 1 
% 2 3 = 1 .9400
% 2 2 1 .9700
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

while abs(L_old - L_new)>0.05
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
        new_cov{1,i} = (A - new_mean(i,:))'*diag(gama(:,i))*(A - new_mean(i,:))/sum(gama(:,i)); 
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
clearvars fileID A data total_data index mean_value data_per_cluster tmp W cov gama term;
fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
A = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
A = A';
train2_data = A;
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

while abs(L_old - L_new)>0.05
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

%validation----------------------------------------------------------------------------------------
fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
A = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
A = A';
total_data = size(A,1);
mean_value = theta{1,1};
cov = theta{1,2};
W = theta{1,3};
L1 = 0;
    for i= 1 : total_data
        tmp = 0;
        for j = 1 : k1
            tmp = tmp + W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        L1 = L1 + log(tmp);
    end
disp(L1);

fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
A = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
A = A';
total_data = size(A,1);
mean_value = theta{2,1};
cov = theta{2,2};
W = theta{2,3};
L2 = 0;
    for i= 1 : total_data
        tmp = 0;
        for j = 1 : k2
            tmp = tmp + W(j)*exp(-0.5*(A(i,:)-mean_value(j,:))*inv(cov{1,j})*(A(i,:)-mean_value(j,:))')/(2*pi*sqrt(det(cov{1,j})));
        end
        L2 = L2 + log(tmp);
    end
disp(L2);

%test-----------------------------------------------------------------------------------------------
clearvars fileID A data total_data index mean_value data_per_cluster tmp W cov gama term;
fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
test1 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
test1 = test1';
test1_data = size(test1,1);
fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
test2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);
test2 = test2';
test2_data = size(test2,1);

test = [test1;test2];
total_test_data = size(test,1);
predicted_class = zeros(total_test_data,1);
k = [k1;k2];

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
            predicted_class(i) = p;
        end
   end
end
predicted_class1 = zeros(size(predicted_class(predicted_class==1),1),2);
predicted_class2 = zeros(size(predicted_class(predicted_class==2),1),2);
end1 =1;end2=1;
for i = 1: size(predicted_class)
   if predicted_class(i)==1
       predicted_class1(end1,:) = test(i,:);
       end1 = end1+1;
   elseif predicted_class(i)==2
       predicted_class2(end2,:) = test(i,:);
       end2 = end2+1; 
   end
end


%%%%%%%%%%%%%%%%%%%%%%%% added by rt

%%%%%%%%%%%%%%%%%%%%%%%%% for train by rt


train_for_cm = [train1_data;train2_data];
total_train_data = size(train_for_cm,1);
predicted_class_for_train = zeros(total_train_data,1);

for i= 1: total_train_data
    max_L_train = -Inf;
   for p = 1 : 2
        tmp_tr = 0;
        for j = 1 : k(p)
            mean_value_tr = theta{p,1}(j,:);
            W = theta{p,3}(j);
            tmp_tr = tmp_tr + W*exp(-0.5*(train_for_cm(i,:)-mean_value_tr)*inv(cov)*(train_for_cm(i,:)-mean_value_tr)')/(2*pi*sqrt(det(cov)));
        end
        if tmp_tr > max_L_train
            max_L_train = tmp_tr;
            predicted_class_for_train(i) = p;
        end
   end
end
end_tr1 =1;end_tr2=1;

for i = 1: size(predicted_class_for_train)
   if predicted_class_for_train(i)==1
       predicted_class_train1(end_tr1,:) = train_for_cm(i,:);
       end_tr1 = end_tr1+1;
   elseif predicted_class_for_train(i)==2
       predicted_class_train2(end_tr2,:) = train_for_cm(i,:);
       end_tr2 = end_tr2+1;
   end 
end
%%%%%%%%%%%%%%%%%%%%%%%% added by rt
count = 0;
confusion_mat_train =  zeros(2,2);
for i = 1 :500
    if(i>= 1 && i<=250)
        if(predicted_class_for_train(i,1) == 1)
            confusion_mat_train(1,1) = confusion_mat_train(1,1)+1 ;
        elseif(predicted_class_for_train(i,1) == 2)
            confusion_mat_train(1,2) = confusion_mat_train(1,2)+1 ;
       % elseif(predicted_class_for_train(i,1) == 3)
        %    confusion_mat_train(1,3) = confusion_mat_train(1,3)+1 ;
        end
    end
    if(i>= 251 && i<=500)
        if(predicted_class_for_train(i,1) == 1)
            confusion_mat_train(2,1) = confusion_mat_train(2,1)+1 ;
        elseif(predicted_class_for_train(i,1) == 2)
            confusion_mat_train(2,2) = confusion_mat_train(2,2)+1 ;
        %elseif(predicted_class_for_train(i,1) == 3)
         %   confusion_mat_train(2,3) = confusion_mat_train(2,3)+1 ;
        end
    end
end

ccount = 0;
confusion_mat_test =  zeros(2,2);
for i = 1 :200
    if(i>= 1 && i<=100)
        if(predicted_class(i,1) == 1)
            confusion_mat_test(1,1) = confusion_mat_test(1,1)+1 ;
        elseif(predicted_class(i,1) == 2)
            confusion_mat_test(1,2) = confusion_mat_test(1,2)+1 ;
        end
    end
    if(i>= 101 && i<=200)
        if(predicted_class(i,1) == 1)
            confusion_mat_test(2,1) = confusion_mat_test(2,1)+1 ;
        elseif(predicted_class(i,1) == 2)
            confusion_mat_test(2,2) = confusion_mat_test(2,2)+1 ;
        end
    end   
end

%%%%%%%%%%%%%

fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
val1 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);

fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
val2 = fscanf(fileID,'%f',[2 Inf]);
fclose(fileID);


val1 = val1';
val2 = val2';
%val3 = val3';

valid_for_cm = [val1;val2];
total_val = size(valid_for_cm,1);
predicted_class_for_val = zeros(total_val,1) ;

for i=1 : total_val
    max_L_val = -Inf;
    for p = 1 : 2
        tmp_val = 0;
        for j = 1 : k(p)
            mean_value_val = theta{p,1}(j,:);
            W = theta{p,3}(j);
           tmp_val = tmp_val + W*exp(-0.5*(valid_for_cm(i,:)-mean_value_val)*inv(cov)*(valid_for_cm(i,:)-mean_value_val)')/(2*pi*sqrt(det(cov)));
        end
        if tmp_val > max_L_val
            max_L_val = tmp_val;
            predicted_class_for_val(i) = p;
        end
   end
end

count = 0;
confusion_mat_val =  zeros(2,2);
for i = 1 :300
    if(i>= 1 && i<=150)
        if(predicted_class_for_val(i,1) == 1)
            confusion_mat_val(1,1) = confusion_mat_val(1,1)+1 ;
        elseif(predicted_class_for_val(i,1) == 2)
            confusion_mat_val(1,2) = confusion_mat_val(1,2)+1 ;
        end
    end
    if(i>= 151 && i<= 300)
        if(predicted_class_for_val(i,1) == 1)
            confusion_mat_val(2,1) = confusion_mat_val(2,1)+1 ;
        elseif(predicted_class_for_val(i,1) == 2)
            confusion_mat_val(2,2) = confusion_mat_val(2,2)+1 ;
        end
    end  
end

%%%%%%%%%%%%%
accuracy_train =( confusion_mat_train(1,1)+confusion_mat_train(2,2))/500  ;
accuracy_test = (confusion_mat_test(1,1)+confusion_mat_test(2,2))/200  ;
accuracy_valid= (confusion_mat_val(1,1)+confusion_mat_val(2,2))/300 ;

%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%

%{
%plot-------------------------------------------------------------------------------
xaxis = -1.7:0.005:2.7;
yaxis = -1.7:0.005:1.5;
[x, y] = meshgrid(xaxis, yaxis);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);

for i= 1: total_plot_data
    max_L = -Inf;
   for p = 1 : 2
        tmp = 0;
        for j = 1 : k(p)
            mean_value = theta{p,1}(j,:);
            cov = theta{p,2}{1,j};
            W = theta{p,3}(j);
            tmp = tmp + W*exp(-0.5*(plot_data(i,:)-mean_value)*inv(cov)*(plot_data(i,:)-mean_value)')/(2*pi*sqrt(det(cov)));
        end
        if tmp > max_L
            max_L = tmp;
            predicted_plot(i) = p;
        end
   end
end

plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');
hold on

scatter(train1_data(:,1),train1_data(:,2),'m','x');
scatter(train2_data(:,1),train2_data(:,2),'r','x');
hold off;
%}
%{
scatter(predicted_class1(:,1),predicted_class1(:,2),'m');
hold on;
scatter(predicted_class2(:,1),predicted_class2(:,2),'g');
scatter(test1(:,1),test1(:,2),'m','.');
scatter(test2(:,1),test2(:,2),'g','.');
hold off;
%}