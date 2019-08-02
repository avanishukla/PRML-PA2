clc;
clearvars;

mydir ='D:\prml\datasets 1_2\datasets 1_2\group8\overlapping\' ;
fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM1 = [A(1:2:end) A(2:2:end)] ;
fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM2 = [A(1:2:end) A(2:2:end)];
fileID = fopen(strcat(mydir,'class3_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM3 = [A(1:2:end) A(2:2:end)];

mean1 = zeros(2,1);mean2 = zeros(2,1);mean3 = zeros(2,1);

for i = 1:250
   mean1(1,1) = mean1(1,1) + AM1(i,1) ; 
   mean1(2,1) = mean1(2,1) + AM1(i,2) ; 
   mean2(1,1) = mean2(1,1) + AM2(i,1) ; 
   mean2(2,1) = mean2(2,1) + AM2(i,2) ; 
   mean3(1,1) = mean3(1,1) + AM3(i,1) ; 
   mean3(2,1) = mean3(2,1) + AM3(i,2) ; 
end

mean1(1,1) = mean1(1,1)/250 ;
mean1(2,1) = mean1(2,1)/250 ;
mean2(1,1) = mean2(1,1)/250 ;
mean2(2,1) = mean2(2,1)/250 ;
mean3(1,1) = mean3(1,1)/250 ;
mean3(2,1) = mean3(2,1)/250 ;

cov_val_c1 = zeros(250,2);
cov_val_c2 = zeros(250,2);
cov_val_c3 = zeros(250,2);

for i= 1:250
    cov_val_c1(i,1) = AM1(i,1)- mean1(1,1) ; 
    cov_val_c1(i,2) = AM1(i,2)- mean1(2,1) ; 
    cov_val_c2(i,1) = AM2(i,1)- mean2(1,1) ; 
    cov_val_c2(i,2) = AM2(i,2)- mean2(2,1) ; 
    cov_val_c3(i,1) = AM3(i,1)- mean3(1,1) ; 
    cov_val_c3(i,2) = AM3(i,2)- mean3(2,1) ;     
end

% cov{1,i} = (data{1,i} - mean_value(i))'*(data{1,i} - mean_value(i))/(data_per_cluster(i)-1);
   
cov_mat1 = (cov_val_c1'*cov_val_c1)/250 ;
cov_mat2 = (cov_val_c2'*cov_val_c2)/250 ;
cov_mat3 = (cov_val_c3'*cov_val_c3)/250 ;

cov_matrix = (cov_mat1+cov_mat2+cov_mat3 )/3 ;
%cov_matrix = cov_mat3 ;
%%%%%%%%%%%%%%%%%
sigma_sq = 1000 ; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% valid
fileID = fopen(strcat(mydir,'\','class1_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val1 = [A(1:2:end) A(2:2:end)] ;
fileID = fopen(strcat(mydir,'/','class2_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val2 = [A(1:2:end) A(2:2:end)];
fileID = fopen(strcat(mydir,'/','class3_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val3 = [A(1:2:end) A(2:2:end)];

probab_val1_c1 = zeros(150,3) ;
probab_val1_c2 = zeros(150,3) ;
probab_val1_c3 = zeros(150,3) ;
x = zeros(2,1);

for i=1:150 
    x(1,1) = val1(i,1)-mean1(1,1);
    x(2,1) = val1(i,2)-mean1(2,1) ;
    probab_val1_c1(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val1(i,1)-mean2(1,1);
    x(2,1) = val1(i,2)-mean2(2,1) ;
    probab_val1_c1(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val1(i,1)-mean3(1,1);
    x(2,1) = val1(i,2)-mean3(2,1) ;
    probab_val1_c1(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
end

calculated_label_val_c1 = zeros(150,1);
count = 0;
confusion_mat_val =  zeros(3,3);


for i = 1 :150
    label_count = max(max(probab_val1_c1(i,1),probab_val1_c1(i,2)),probab_val1_c1(i,3));
    if (probab_val1_c1(i,1) == probab_val1_c1(i,2) )
        calculated_label_val_c1(i,1) = 1;
    elseif label_count == probab_val1_c1(i,1)
       calculated_label_val_c1(i,1) = 1;
       confusion_mat_val(1,1) = confusion_mat_val(1,1)+1;
    elseif label_count == probab_val1_c1(i,2)
        calculated_label_val_c1(i,1) = 2;
        confusion_mat_val(1,2) = confusion_mat_val(1,2)+1;
    else
        calculated_label_val_c1(i,1) = 3;
        confusion_mat_val(1,3) = confusion_mat_val(1,3)+1;
    end
     
     if(calculated_label_val_c1(i,1) ~= 1)
         count = count+1;
     end 
end


for i = 1:150
 %valid 2
    x(1,1) = val2(i,1)-mean1(1,1);
    x(2,1) = val2(i,2)-mean1(2,1) ;
    probab_val1_c2(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val2(i,1)-mean2(1,1);
    x(2,1) = val2(i,2)-mean2(2,1) ;
    probab_val1_c2(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val2(i,1)-mean3(1,1);
    x(2,1) = val2(i,2)-mean3(2,1) ;
    probab_val1_c2(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
end

calculated_label_val_c2 = zeros(150,1) ;
for i = 1 :150
    label_count = max(max(probab_val1_c2(i,1),probab_val1_c2(i,2)),probab_val1_c2(i,3));
    if (probab_val1_c2(i,1) == probab_val1_c2(i,2) )
        calculated_label_val_c2(i,1) = 1;
    elseif label_count == probab_val1_c2(i,1)
       calculated_label_val_c2(i,1) = 1;
       confusion_mat_val(2,1) = confusion_mat_val(2,1)+1;
    elseif label_count == probab_val1_c2(i,2)
        calculated_label_val_c2(i,1) = 2;
        confusion_mat_val(2,2) = confusion_mat_val(2,2)+1;
    else
        calculated_label_val_c2(i,1) = 3;
        confusion_mat_val(2,3) = confusion_mat_val(2,3)+1;
    end
     if(calculated_label_val_c2(i,1) ~= 2)
         count = count+1;
     end 
end

for i = 1:150
 %valid 2
    x(1,1) = val3(i,1)-mean1(1,1);
    x(2,1) = val3(i,2)-mean1(2,1) ;
    probab_val1_c3(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val3(i,1)-mean2(1,1);
    x(2,1) = val3(i,2)-mean2(2,1) ;
    probab_val1_c3(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = val3(i,1)-mean3(1,1);
    x(2,1) = val3(i,2)-mean3(2,1) ;
    probab_val1_c3(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
end
disp(count);
count = 0;
calculated_label_val_c3 = zeros(150,1) ;
for i = 1 :150
    label_count = max(max(probab_val1_c3(i,1),probab_val1_c3(i,2)),probab_val1_c3(i,3));
    if (probab_val1_c3(i,1) == probab_val1_c3(i,2) )
        calculated_label_val_c3(i,1) = 1;
    elseif label_count == probab_val1_c3(i,1)
       calculated_label_val_c3(i,1) = 1;
       confusion_mat_val(3,1) = confusion_mat_val(3,1)+1;
    elseif label_count == probab_val1_c3(i,2)
        calculated_label_val_c3(i,1) = 2;
        confusion_mat_val(3,2) = confusion_mat_val(3,2)+1;
    else
        calculated_label_val_c3(i,1) = 3;
        confusion_mat_val(3,3) = confusion_mat_val(3,3)+1;
    end
     if(calculated_label_val_c3(i,1) ~= 3)
         count = count+1;
     end 
end
disp(count);


fileID = fopen(strcat(mydir,'\','class1_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c1 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'\','class2_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c2 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'\','class3_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c3 = [A(1:2:end) A(2:2:end)] ;

x = zeros(2,1) ;
probab_test1_given_class = zeros(100,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt
for i = 1:100
    x(1,1) = test_c1(i,1)-mean1(1,1);
    x(2,1) = test_c1(i,2)-mean1(2,1) ;
    probab_test1_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c1(i,1)-mean2(1,1);
    x(2,1) = test_c1(i,2)-mean2(2,1) ;
    probab_test1_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c1(i,1)-mean3(1,1);
    x(2,1) = test_c1(i,2)-mean3(2,1) ;
    probab_test1_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
   
end

calculated_label_test_c1 = zeros(100,1);
confusion_mat_test =  zeros(3,3);
for i = 1 :100
    label_count = max(max(probab_test1_given_class(i,1),probab_test1_given_class(i,2)),probab_test1_given_class(i,3));
    if (probab_test1_given_class(i,1) == probab_test1_given_class(i,2) )
        calculated_label_test_c1(i,1) = 1;
    elseif label_count == probab_test1_given_class(i,1)
       calculated_label_test_c1(i,1) = 1;
       confusion_mat_test(1,1) = confusion_mat_test(1,1)+1;
    elseif label_count == probab_test1_given_class(i,2)
        calculated_label_test_c1(i,1) = 2;
        confusion_mat_test(1,2) = confusion_mat_test(1,2)+1;
    else
        calculated_label_test_c1(i,1) = 3;
        confusion_mat_test(1,3) = confusion_mat_test(1,3)+1;
    end
     if(calculated_label_test_c1(i,1) ~= 1)
         count = count+1;
     end 
end
disp(count) ;


probab_test2_given_class = zeros(100,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt
for i = 1:100
    x(1,1) = test_c2(i,1)-mean1(1,1);
    x(2,1) = test_c2(i,2)-mean1(2,1) ;
    probab_test2_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c2(i,1)-mean2(1,1);
    x(2,1) = test_c2(i,2)-mean2(2,1) ;
    probab_test2_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c2(i,1)-mean3(1,1);
    x(2,1) = test_c2(i,2)-mean3(2,1) ;
    probab_test2_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );   
end

calculated_label_test_c2 = zeros(100,1);
for i = 1 :100
    label_count = max(max(probab_test2_given_class(i,1),probab_test2_given_class(i,2)),probab_test2_given_class(i,3));
    if (probab_test2_given_class(i,1) == probab_test2_given_class(i,2) )
        calculated_label_test_c2(i,1) = 1;
    elseif label_count == probab_test2_given_class(i,1)
       calculated_label_test_c2(i,1) = 1;
       confusion_mat_test(2,1) = confusion_mat_test(2,1)+1;
    elseif label_count == probab_test2_given_class(i,2)
        calculated_label_test_c2(i,1) = 2;
        confusion_mat_test(2,2) = confusion_mat_test(2,2)+1;
    else
        calculated_label_test_c2(i,1) = 3;
        confusion_mat_test(2,3) = confusion_mat_test(2,3)+1;
    end
     if(calculated_label_test_c2(i,1) ~= 2)
         count = count+1;
     end 
end
disp(count) ;



probab_test3_given_class = zeros(100,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt
for i = 1:100
    x(1,1) = test_c3(i,1)-mean1(1,1);
    x(2,1) = test_c3(i,2)-mean1(2,1) ;
    probab_test3_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c3(i,1)-mean2(1,1);
    x(2,1) = test_c3(i,2)-mean2(2,1) ;
    probab_test3_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = test_c3(i,1)-mean3(1,1);
    x(2,1) = test_c3(i,2)-mean3(2,1) ;
    probab_test3_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
   
end

calculated_label_test_c3 = zeros(100,1);

for i = 1 :100
    label_count = max(max(probab_test3_given_class(i,1),probab_test3_given_class(i,2)),probab_test3_given_class(i,3));
    if (probab_test3_given_class(i,1) == probab_test3_given_class(i,2) )
        calculated_label_test_c3(i,1) = 1;
    elseif label_count == probab_test3_given_class(i,1)
       calculated_label_test_c3(i,1) = 1;
       confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == probab_test3_given_class(i,2)
        calculated_label_test_c3(i,1) = 2;
        confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
    else
        calculated_label_test_c3(i,1) = 3;
        confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
     if(calculated_label_test_c3(i,1) ~= 3)
         count = count+1;
     end 
end
disp(count) ;



%%%%%%%%%%%%%%%%% training error train 111111111111111111

probab_train1_given_class = zeros(250,1);
confusion_mat_train = zeros(3,3);
for i = 1:250
    x(1,1) = AM1(i,1)-mean1(1,1);
    x(2,1) = AM1(i,2)-mean1(2,1) ;
    probab_train1_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM1(i,1)-mean2(1,1);
    x(2,1) = AM1(i,2)-mean2(2,1) ;
    probab_train1_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM1(i,1)-mean3(1,1);
    x(2,1) = AM1(i,2)-mean3(2,1) ;
    probab_train1_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );   
end

calculated_label_train_c1 = zeros(250,1);
for i = 1 :250
    label_count = max(max(probab_train1_given_class(i,1),probab_train1_given_class(i,2)),probab_train1_given_class(i,3));
    if (probab_train1_given_class(i,1) == probab_train1_given_class(i,2) )
        calculated_label_train_c1(i,1) = 1;
    elseif label_count == probab_train1_given_class(i,1)
       calculated_label_train_c1(i,1) = 1;
       confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
    elseif label_count == probab_train1_given_class(i,2)
        calculated_label_train_c1(i,1) = 2;
        confusion_mat_train(1,2) = confusion_mat_train(1,2)+1;
    else
        calculated_label_train_c1(i,1) = 3;
        confusion_mat_train(1,3) = confusion_mat_train(1,3)+1;
    end
     if(calculated_label_train_c1(i,1) ~= 1)
         count = count+1;
     end 
end
disp(count) ;


%%%%%%%%%%%%%%%%% training error

probab_train2_given_class = zeros(250,1);

for i = 1:250
    x(1,1) = AM2(i,1)-mean1(1,1);
    x(2,1) = AM2(i,2)-mean1(2,1) ;
    probab_train2_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM2(i,1)-mean2(1,1);
    x(2,1) = AM2(i,2)-mean2(2,1) ;
    probab_train2_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM2(i,1)-mean3(1,1);
    x(2,1) = AM2(i,2)-mean3(2,1) ;
    probab_train2_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );   
end

calculated_label_train_c2 = zeros(250,1);
for i = 1 :250
    label_count = max(max(probab_train2_given_class(i,1),probab_train2_given_class(i,2)),probab_train2_given_class(i,3));
    if (probab_train2_given_class(i,1) == probab_train2_given_class(i,2) )
        calculated_label_train_c2(i,1) = 1;
    elseif label_count == probab_train2_given_class(i,1)
       calculated_label_train_c2(i,1) = 1;
       confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
    elseif label_count == probab_train2_given_class(i,2)
        calculated_label_train_c2(i,1) = 2;
        confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
    else
        calculated_label_train_c2(i,1) = 3;
        confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
    end
     if(calculated_label_train_c2(i,1) ~= 2)
         count = count+1;
     end 
end
disp(count) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% train 3

probab_train3_given_class = zeros(250,3);


for i = 1:250
    x(1,1) = AM3(i,1)-mean1(1,1);
    x(2,1) = AM3(i,2)-mean1(2,1) ;
    probab_train3_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM3(i,1)-mean2(1,1);
    x(2,1) = AM3(i,2)-mean2(2,1) ;
    probab_train3_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
    
    x(1,1) = AM3(i,1)-mean3(1,1);
    x(2,1) = AM3(i,2)-mean3(2,1) ;
    probab_train3_given_class(i,3)= (exp(-0.5 * x'*inv(cov_matrix)*x ) / sqrt(det(cov_matrix)) );
   
end

calculated_label_train_c3 = zeros(250,1);

for i = 1 :250
    label_count = max(max(probab_train3_given_class(i,1),probab_train3_given_class(i,2)),probab_train3_given_class(i,3));
    if (probab_train3_given_class(i,1) == probab_train3_given_class(i,2) )
        calculated_label_train_c3(i,1) = 1;
    elseif label_count == probab_train3_given_class(i,1)
       calculated_label_train_c3(i,1) = 1;
       confusion_mat_train(3,1) = confusion_mat_train(3,1)+1;
    elseif label_count == probab_train3_given_class(i,2)
        calculated_label_train_c3(i,1) = 2;
        confusion_mat_train(3,2) = confusion_mat_train(3,2)+1;
    else
        calculated_label_train_c3(i,1) = 3;
        confusion_mat_train(3,3) = confusion_mat_train(3,3)+1;
    end
     if(calculated_label_train_c3(i,1) ~= 3)
         count = count+1;
     end 
end
disp(count) ;
accuracy_train =( confusion_mat_train(1,1)+confusion_mat_train(2,2)+confusion_mat_train(3,3))/750  ;
accuracy_test = (confusion_mat_test(1,1)+confusion_mat_test(2,2)+confusion_mat_test(3,3))/300  ;
accuracy_valid= (confusion_mat_val(1,1)+confusion_mat_val(2,2)+confusion_mat_val(3,3))/450 ;
%%%%%%%%%%%%

%%%%%%%%%%%%%%%%plot
%%%%%%%%%%%%%%%%plot
xaxis = -10:0.04:15;
yaxis = -15:0.04:15;
[x, y] = meshgrid(xaxis, yaxis);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);

for i= 1: total_plot_data
    i
    x1(1,1) = plot_data(i,1)-mean1(1,1);
    x1(2,1) = plot_data(i,2)-mean1(2,1) ;
    probab_plot(i,1)= (exp(-0.5 * x1'*inv(cov_matrix)*x1 ) / sqrt(det(cov_matrix)) );
    
    x1(1,1) = plot_data(i,1)-mean2(1,1);
    x1(2,1) = plot_data(i,2)-mean2(2,1) ;
    probab_plot(i,2)= (exp(-0.5 * x1'*inv(cov_matrix)*x1 ) / sqrt(det(cov_matrix)) );
    
    x1(1,1) = plot_data(i,1)-mean3(1,1);
    x1(2,1) = plot_data(i,2)-mean3(2,1) ;
    probab_plot(i,3)= (exp(-0.5 * x1'*inv(cov_matrix)*x1 ) / sqrt(det(cov_matrix)) );
   
    label_count = max(max(probab_plot(i,1),probab_plot(i,2)),probab_plot(i,3));
    if (probab_plot(i,1) == probab_plot(i,2) )
        predicted_plot(i,1) = 1;
    elseif label_count == probab_plot(i,1)
       predicted_plot(i,1) = 1;
    %   confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == probab_plot(i,2)
        predicted_plot(i,1) = 2;
      %  confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
    else
        predicted_plot(i,1) = 3;
       % confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
     if(predicted_plot(i,1) ~= 3)
         count = count+1;
     end 
    
end
%for i = 1 :total_plot_data
 %   i
    
%end
plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');

hold on;
scatter(AM1(:,1),AM1(:,2),'m','x');
scatter(AM2(:,1),AM2(:,2),'w','x');
scatter(AM3(:,1),AM3(:,2),'r','x');
xlabel('x1');
ylabel('x2');
str  = { strcat('Decision Region Plot using Bayes classifier with same covariance matrix on overlapping data')};
title(str,'FontSize',15);
t = text(-5,10,'CLASS 1','Color','r','FontSize',14);
t1 = text(5,15,'CLASS 2','Color','b','FontSize',14);
t3 = text(5,-15,'CLASS 3','Color','b','FontSize',14);

hold off;
