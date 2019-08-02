clc;
clearvars;

mydir ='D:\prml\datasets 1_2\datasets 1_2\group8\nonlinearly_separable' ;
fileID = fopen(strcat(mydir,'class1_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM1 = [A(1:2:end) A(2:2:end)] ;
fileID = fopen(strcat(mydir,'class2_train.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM2 = [A(1:2:end) A(2:2:end)];

mean1 = zeros(2,1);mean2 = zeros(2,1);
for i = 1:250
   mean1(1,1) = mean1(1,1) + AM1(i,1) ; 
   mean1(2,1) = mean1(2,1) + AM1(i,2) ; 
   mean2(1,1) = mean2(1,1) + AM2(i,1) ; 
   mean2(2,1) = mean2(2,1) + AM2(i,2) ;  
end

mean1(1,1) = mean1(1,1)/250 ;
mean1(2,1) = mean1(2,1)/250 ;
mean2(1,1) = mean2(1,1)/250 ;
mean2(2,1) = mean2(2,1)/250 ;

cov_val_c1 = zeros(250,2);
cov_val_c2 = zeros(250,2);

for i= 1:250
    cov_val_c1(i,1) = AM1(i,1)- mean1(1,1) ; 
    cov_val_c1(i,2) = AM1(i,2)- mean1(2,1) ; 
    cov_val_c2(i,1) = AM2(i,1)- mean2(1,1) ; 
    cov_val_c2(i,2) = AM2(i,2)- mean2(2,1) ; 
end
   
cov_mat1 = (cov_val_c1'*cov_val_c1)/250 ;
cov_mat2 = (cov_val_c2'*cov_val_c2)/250 ;

cov_matrix_with_off_dia_zero1 = zeros(2,2);
cov_matrix_with_off_dia_zero2 = zeros(2,2);

for i = 1:2 
    for j = 1:2 
    if i==j
        cov_matrix_with_off_dia_zero1(i,j) = cov_mat1(i,j);
        cov_matrix_with_off_dia_zero2(i,j) = cov_mat2(i,j);
    end
    end
end

%%%%%%%%%%%%% valid
fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);

val1 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);

val2 = [A(1:2:end) A(2:2:end)];

probab_val1_c1 = zeros(150,2) ;
probab_val1_c2 = zeros(150,2) ;

x = zeros(2,1);

for i=1:150 
    x(1,1) = val1(i,1)-mean1(1,1);
    x(2,1) = val1(i,2)-mean1(2,1) ;
    probab_val1_c1(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = val1(i,1)-mean2(1,1);
    x(2,1) = val1(i,2)-mean2(2,1) ;
    probab_val1_c1(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
    
end

calculated_label_val_c1 = zeros(150,1);
count = 0;
confusion_mat_val =  zeros(2,2);

for i = 1 :150
    label_count = max(probab_val1_c1(i,1),probab_val1_c1(i,2));
    if (probab_val1_c1(i,1) == probab_val1_c1(i,2) )
        calculated_label_val_c1(i,1) = 1;
    elseif label_count == probab_val1_c1(i,1)
       calculated_label_val_c1(i,1) = 1;
       confusion_mat_val(1,1) = confusion_mat_val(1,1)+1;
    elseif label_count == probab_val1_c1(i,2)
        calculated_label_val_c1(i,1) = 2;
        confusion_mat_val(1,2) = confusion_mat_val(1,2)+1;
    end
end
%%%%%%%%%%%%%%%55

for i = 1:150
 %valid 2
    x(1,1) = val2(i,1)-mean1(1,1);
    x(2,1) = val2(i,2)-mean1(2,1) ;
    probab_val1_c2(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = val2(i,1)-mean2(1,1);
    x(2,1) = val2(i,2)-mean2(2,1) ;
    probab_val1_c2(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
    
end

calculated_label_val_c2 = zeros(150,1) ;
for i = 1 :150
    label_count = max(probab_val1_c2(i,1),probab_val1_c2(i,2));
    if (probab_val1_c2(i,1) == probab_val1_c2(i,2) )
        calculated_label_val_c2(i,1) = 1;
    elseif label_count == probab_val1_c2(i,1)
       calculated_label_val_c2(i,1) = 1;
       confusion_mat_val(2,1) = confusion_mat_val(2,1)+1;
    elseif label_count == probab_val1_c2(i,2)
        calculated_label_val_c2(i,1) = 2;
        confusion_mat_val(2,2) = confusion_mat_val(2,2)+1;
    end
end

fileID = fopen(strcat(mydir,'class1_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c1 = [A(1:2:end) A(2:2:end)] ;

fileID = fopen(strcat(mydir,'class2_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c2 = [A(1:2:end) A(2:2:end)] ;

x = zeros(2,1) ;
probab_test1_given_class = zeros(100,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt 1
for i = 1:100
    x(1,1) = test_c1(i,1)-mean1(1,1);
    x(2,1) = test_c1(i,2)-mean1(2,1) ;
    probab_test1_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = test_c1(i,1)-mean2(1,1);
    x(2,1) = test_c1(i,2)-mean2(2,1) ;
    probab_test1_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
   
end

calculated_label_test_c1 = zeros(100,1);
confusion_mat_test =  zeros(2,2);
for i = 1 :100 
    label_count = max(probab_test1_given_class(i,1),probab_test1_given_class(i,2)) ; 
    if (probab_test1_given_class(i,1) == probab_test1_given_class(i,2) )
        calculated_label_test_c1(i,1) = 1;
    elseif label_count == probab_test1_given_class(i,1)
       calculated_label_test_c1(i,1) = 1;
       confusion_mat_test(1,1) = confusion_mat_test(1,1)+1;
    elseif label_count == probab_test1_given_class(i,2)
        calculated_label_test_c1(i,1) = 2;
        confusion_mat_test(1,2) = confusion_mat_test(1,2)+1;
    end
end

probab_test2_given_class = zeros(100,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt class 2
for i = 1:100
    x(1,1) = test_c2(i,1)-mean1(1,1);
    x(2,1) = test_c2(i,2)-mean1(2,1) ;
    probab_test2_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = test_c2(i,1)-mean2(1,1);
    x(2,1) = test_c2(i,2)-mean2(2,1) ;
    probab_test2_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
    
end

calculated_label_test_c2 = zeros(100,1);
for i = 1 :100
    label_count = max(probab_test2_given_class(i,1),probab_test2_given_class(i,2));
    if (probab_test2_given_class(i,1) == probab_test2_given_class(i,2) )
        calculated_label_test_c2(i,1) = 1;
    elseif label_count == probab_test2_given_class(i,1)
       calculated_label_test_c2(i,1) = 1;
       confusion_mat_test(2,1) = confusion_mat_test(2,1)+1;
    elseif label_count == probab_test2_given_class(i,2)
        calculated_label_test_c2(i,1) = 2;
        confusion_mat_test(2,2) = confusion_mat_test(2,2)+1;
     end 
end
disp(count) ;



%%%%%%%%%%%%%%%%% training error train 111111111111111111

probab_train1_given_class = zeros(250,1);
confusion_mat_train = zeros(2,2);
for i = 1:250
    x(1,1) = AM1(i,1)-mean1(1,1);
    x(2,1) = AM1(i,2)-mean1(2,1) ;
    probab_train1_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = AM1(i,1)-mean2(1,1);
    x(2,1) = AM1(i,2)-mean2(2,1) ;
    probab_train1_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
     
end

calculated_label_train_c1 = zeros(250,1);
for i = 1 :250
    label_count = max(probab_train1_given_class(i,1),probab_train1_given_class(i,2));
    if (probab_train1_given_class(i,1) == probab_train1_given_class(i,2) )
        calculated_label_train_c1(i,1) = 1;
    elseif label_count == probab_train1_given_class(i,1)
       calculated_label_train_c1(i,1) = 1;
       confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
    elseif label_count == probab_train1_given_class(i,2)
        calculated_label_train_c1(i,1) = 2;
        confusion_mat_train(1,2) = confusion_mat_train(1,2)+1;
    else
     %   calculated_label_train_c1(i,1) = 3;
    %    confusion_mat_train(1,3) = confusion_mat_train(1,3)+1;
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
    probab_train2_given_class(i,1)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero1)*x ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x(1,1) = AM2(i,1)-mean2(1,1);
    x(2,1) = AM2(i,2)-mean2(2,1) ;
    probab_train2_given_class(i,2)= (exp(-0.5 * x'*inv(cov_matrix_with_off_dia_zero2)*x ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
    
end

calculated_label_train_c2 = zeros(250,1);
for i = 1 :250
    label_count = max(probab_train2_given_class(i,1),probab_train2_given_class(i,2));
    if (probab_train2_given_class(i,1) == probab_train2_given_class(i,2) )
        calculated_label_train_c2(i,1) = 1;
    elseif label_count == probab_train2_given_class(i,1)
       calculated_label_train_c2(i,1) = 1;
       confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
    elseif label_count == probab_train2_given_class(i,2)
        calculated_label_train_c2(i,1) = 2;
        confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
    else
    %    calculated_label_train_c2(i,1) = 3;
     %   confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
    end
     if(calculated_label_train_c2(i,1) ~= 2)
         count = count+1;
     end 
end
disp(count) ;

accuracy_train =( confusion_mat_train(1,1)+confusion_mat_train(2,2))/500  ;
accuracy_test = (confusion_mat_test(1,1)+confusion_mat_test(2,2))/200  ;
accuracy_valid= (confusion_mat_val(1,1)+confusion_mat_val(2,2))/300 ;

%%%%%%%%%%%%%%%%plot
%%%%%%%%%%%%%%%%plot
%{
xaxis = -1.7:0.005:2.7;
yaxis = -1.7:0.005:1.5;
[x, y] = meshgrid(xaxis, yaxis);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);

for i=1:total_plot_data 
    i
    x1(1,1) = plot_data(i,1)-mean1(1,1);
    x1(2,1) = plot_data(i,2)-mean1(2,1) ;
    probab_plot(i,1)= (exp(-0.5 * x1'*inv(cov_matrix_with_off_dia_zero1)*x1 ) / sqrt(det(cov_matrix_with_off_dia_zero1)) );
    
    x1(1,1) = plot_data(i,1)-mean2(1,1);
    x1(2,1) = plot_data(i,2)-mean2(2,1) ;
    probab_plot(i,2)= (exp(-0.5 * x1'*inv(cov_matrix_with_off_dia_zero2)*x1 ) / sqrt(det(cov_matrix_with_off_dia_zero2)) );
    
    label_count = max(probab_plot(i,1),probab_plot(i,2));
    if (probab_plot(i,1) == probab_plot(i,2) )
        predicted_plot(i,1) = 1;
    elseif label_count == probab_plot(i,1)
       predicted_plot(i,1) = 1;
    elseif label_count == probab_plot(i,2)
        predicted_plot(i,1) = 2;
    end 
end

plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');

hold on;
scatter(AM1(:,1),AM1(:,2),'m','x');
scatter(AM2(:,1),AM2(:,2),'w','x');

xlabel('x1')
ylabel('x2')

str  = { strcat('Decision Region Plot naive bayes diff cov non linear')};
title(str,'FontSize',15);

t = text(-5,10,'CLASS 1','Color','r','FontSize',14);
t1 = text(0,10,'CLASS 2','Color','b','FontSize',14);

hold off;


%}