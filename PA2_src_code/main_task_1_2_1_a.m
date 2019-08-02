clc;
clearvars;

mydir ='D:\prml\datasets 1_2\datasets 1_2\group8\linearly_separable\' ;
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

fileID = fopen(strcat(mydir,'class1_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val_c1 = [A(1:2:end) A(2:2:end)] ;
fileID = fopen(strcat(mydir,'class2_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val_c2 = [A(1:2:end) A(2:2:end)];
fileID = fopen(strcat(mydir,'class3_val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
val_c3 = [A(1:2:end) A(2:2:end)];

%%%%%%%%%%%%%%%%
sigma_square = 1000 ;
%%%%%%%%%%%

p_data_val1_given_cls = zeros(150,6);

for i = 1:150
    p_data_val1_given_cls(i,1) = exp(-((power(val_c1(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val1_given_cls(i,2) = exp(-((power(val_c1(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val1_given_cls(i,3) = exp(-((power(val_c1(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val1_given_cls(i,4) = exp(-((power(val_c1(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val1_given_cls(i,5) = exp(-((power(val_c1(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val1_given_cls(i,6) = exp(-((power(val_c1(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_val1 = zeros(150,3);

for i= 1:150
    ans_val1(i,1) =   (p_data_val1_given_cls(i,1)*p_data_val1_given_cls(i,2)) /3 ;
    ans_val1(i,2)=   (p_data_val1_given_cls(i,3)*p_data_val1_given_cls(i,4)) /3 ;
    ans_val1(i,3)=   (p_data_val1_given_cls(i,5)*p_data_val1_given_cls(i,6)) /3 ;
end

calculated_label_val_c1 = zeros(150,1);
count = 0;
confusion_mat_val =  zeros(3,3);

for i = 1 :150
    label_count = max(max(ans_val1(i,1),ans_val1(i,2)),ans_val1(i,3));
    if (ans_val1(i,1) == ans_val1(i,2) )
        calculated_label_val_c1(i,1) = 1;
    elseif label_count == ans_val1(i,1)
       calculated_label_val_c1(i,1) = 1;
       confusion_mat_val(1,1) = confusion_mat_val(1,1)+1;
    elseif label_count == ans_val1(i,2)
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

p_data_val2_given_cls = zeros(150,6);
for i = 1:150
    p_data_val2_given_cls(i,1) = exp(-((power(val_c2(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val2_given_cls(i,2) = exp(-((power(val_c2(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val2_given_cls(i,3) = exp(-((power(val_c2(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val2_given_cls(i,4) = exp(-((power(val_c2(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val2_given_cls(i,5) = exp(-((power(val_c2(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val2_given_cls(i,6) = exp(-((power(val_c2(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_val2 = zeros(150,3);

for i= 1:150
    ans_val2(i,1) =  (p_data_val2_given_cls(i,1)*p_data_val2_given_cls(i,2)) /3 ;
    ans_val2(i,2)=   (p_data_val2_given_cls(i,3)*p_data_val2_given_cls(i,4)) /3 ;
    ans_val2(i,3)=   (p_data_val2_given_cls(i,5)*p_data_val2_given_cls(i,6)) /3 ;
end

calculated_label_val_c2 = zeros(150,1) ;
for i = 1 :150
    label_count = max(max(ans_val2(i,1),ans_val2(i,2)),ans_val2(i,3));
    if (ans_val2(i,1) == ans_val2(i,2) )
        calculated_label_val_c2(i,1) = 1;
    elseif label_count == ans_val2(i,1)
       calculated_label_val_c2(i,1) = 1;
       confusion_mat_val(2,1) = confusion_mat_val(2,1)+1;
    elseif label_count == ans_val2(i,2)
        calculated_label_val_c2(i,1) = 2;
        confusion_mat_val(2,2) = confusion_mat_val(2,2)+1;
    else
        calculated_label_val_c2(i,1) = 3;
        confusion_mat_val(2,3) = confusion_mat_val(2,3)+1;
    end
     if(calculated_label_val_c1(i,1) ~= 1)
         count = count+1;
     end 
end

p_data_val3_given_cls = zeros(150,6);
for i = 1:150
    p_data_val3_given_cls(i,1) = exp(-((power(val_c3(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val3_given_cls(i,2) = exp(-((power(val_c3(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val3_given_cls(i,3) = exp(-((power(val_c3(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val3_given_cls(i,4) = exp(-((power(val_c3(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_val3_given_cls(i,5) = exp(-((power(val_c3(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_val3_given_cls(i,6) = exp(-((power(val_c3(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_val3 = zeros(150,3);

for i= 1:150
    ans_val3(i,1) =  (p_data_val3_given_cls(i,1)*p_data_val3_given_cls(i,2)) /3 ;
    ans_val3(i,2)=   (p_data_val3_given_cls(i,3)*p_data_val3_given_cls(i,4)) /3 ;
    ans_val3(i,3)=   (p_data_val3_given_cls(i,5)*p_data_val3_given_cls(i,6)) /3 ;
end

calculated_label_val_c3 = zeros(150,1) ;
for i = 1 :150
    label_count = max(max(ans_val3(i,1),ans_val3(i,2)),ans_val3(i,3));
    if (ans_val3(i,1) == ans_val3(i,2) )
        calculated_label_val_c3(i,1) = 1;
    elseif label_count == ans_val3(i,1)
       calculated_label_val_c3(i,1) = 1;
       confusion_mat_val(3,1) = confusion_mat_val(3,1)+1;
    elseif label_count == ans_val3(i,2)
        calculated_label_val_c3(i,1) = 2;
        confusion_mat_val(3,2) = confusion_mat_val(3,2)+1;
    else
        calculated_label_val_c3(i,1) = 3;
        confusion_mat_val(3,3) = confusion_mat_val(3,3)+1;
    end
     if(calculated_label_val_c1(i,1) ~= 1)
         count = count+1;
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

fileID = fopen(strcat(mydir,'class3_test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
test_c3 = [A(1:2:end) A(2:2:end)] ;

p_data_test1_given_cls = zeros(150,6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt
for i = 1:100
    p_data_test1_given_cls(i,1) = exp(-((power(test_c1(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test1_given_cls(i,2) = exp(-((power(test_c1(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test1_given_cls(i,3) = exp(-((power(test_c1(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test1_given_cls(i,4) = exp(-((power(test_c1(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test1_given_cls(i,5) = exp(-((power(test_c1(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test1_given_cls(i,6) = exp(-((power(test_c1(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_test1 = zeros(100,3);

for i= 1:100
    ans_test1(i,1) =   (p_data_test1_given_cls(i,1)*p_data_test1_given_cls(i,2)) /3 ;
    ans_test1(i,2)=   (p_data_test1_given_cls(i,3)*p_data_test1_given_cls(i,4)) /3 ;
    ans_test1(i,3)=   (p_data_test1_given_cls(i,5)*p_data_test1_given_cls(i,6)) /3 ;
end

calculated_label_test_c1 = zeros(100,1);
confusion_mat_test =  zeros(3,3);
for i = 1 :100
    label_count = max(max(ans_test1(i,1),ans_test1(i,2)),ans_test1(i,3));
    if (ans_test1(i,1) == ans_test1(i,2) )
        calculated_label_test_c1(i,1) = 1;
    elseif label_count == ans_test1(i,1)
       calculated_label_test_c1(i,1) = 1;
       confusion_mat_test(1,1) = confusion_mat_test(1,1)+1;
    elseif label_count == ans_test1(i,2)
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

p_data_test2_given_cls = zeros(100,6);
for i = 1:100
    p_data_test2_given_cls(i,1) = exp(-((power(test_c2(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test2_given_cls(i,2) = exp(-((power(test_c2(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test2_given_cls(i,3) = exp(-((power(test_c2(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test2_given_cls(i,4) = exp(-((power(test_c2(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test2_given_cls(i,5) = exp(-((power(test_c2(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test2_given_cls(i,6) = exp(-((power(test_c2(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_test2 = zeros(100,3);

for i= 1:100
    ans_test2(i,1) =  (p_data_test2_given_cls(i,1)*p_data_test2_given_cls(i,2)) /3 ;
    ans_test2(i,2)=   (p_data_test2_given_cls(i,3)*p_data_test2_given_cls(i,4)) /3 ;
    ans_test2(i,3)=   (p_data_test2_given_cls(i,5)*p_data_test2_given_cls(i,6)) /3 ;
end
calculated_label_test_c2 = zeros(100,1) ;
for i = 1 :100
    label_count = max(max(ans_test2(i,1),ans_test2(i,2)),ans_test2(i,3));
    if (ans_test2(i,1) == ans_test2(i,2) )
        calculated_label_test_c2(i,1) = 1;
    elseif label_count == ans_test2(i,1)
       calculated_label_test_c2(i,1) = 1;
       confusion_mat_test(2,1) = confusion_mat_test(2,1)+1;
    elseif label_count == ans_test2(i,2)
        calculated_label_test_c2(i,1) = 2;
        confusion_mat_test(2,2) = confusion_mat_test(2,2)+1;
    else
        calculated_label_test_c2(i,1) = 3;
        confusion_mat_test(2,3) = confusion_mat_test(2,3)+1;
    end
     if(calculated_label_test_c2(i,1) ~= 1)
         count = count+1;
     end 
end

p_data_test3_given_cls = zeros(100,6);
for i = 1:100
    p_data_test3_given_cls(i,1) = exp(-((power(test_c3(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test3_given_cls(i,2) = exp(-((power(test_c3(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test3_given_cls(i,3) = exp(-((power(test_c3(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test3_given_cls(i,4) = exp(-((power(test_c3(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_test3_given_cls(i,5) = exp(-((power(test_c3(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_test3_given_cls(i,6) = exp(-((power(test_c3(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_test3 = zeros(100,3);

for i= 1:100
    ans_test3(i,1) =  (p_data_test3_given_cls(i,1)*p_data_test3_given_cls(i,2)) /3 ;
    ans_test3(i,2)=   (p_data_test3_given_cls(i,3)*p_data_test3_given_cls(i,4)) /3 ;
    ans_test3(i,3)=   (p_data_test3_given_cls(i,5)*p_data_test3_given_cls(i,6)) /3 ;
end

calculated_label_test_c3 = zeros(100,1) ;
for i = 1 :100
    label_count = max(max(ans_test3(i,1),ans_test3(i,2)),ans_test3(i,3));
    if (ans_test3(i,1) == ans_test3(i,2) )
        calculated_label_test_c3(i,1) = 1;
    elseif label_count == ans_test3(i,1)
       calculated_label_test_c3(i,1) = 1;
       confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == ans_test3(i,2)
        calculated_label_test_c3(i,1) = 2;
        confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
    else
        calculated_label_test_c3(i,1) = 3;
        confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
     if(calculated_label_test_c3(i,1) ~= 1)
         count = count+1;
     end 
end


%%%%%%%%%%%%%%%%%%%%% traiiiiiiiiiiiiiiiin error cal


p_data_train1_given_cls = zeros(250,6);
count = 0 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testtttttttttttttttttt
for i = 1:250
    p_data_train1_given_cls(i,1) = exp(-((power(AM1(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train1_given_cls(i,2) = exp(-((power(AM1(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train1_given_cls(i,3) = exp(-((power(AM1(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train1_given_cls(i,4) = exp(-((power(AM1(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train1_given_cls(i,5) = exp(-((power(AM1(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train1_given_cls(i,6) = exp(-((power(AM1(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_train1 = zeros(250,3);

for i= 1:250
    ans_train1(i,1) =  (p_data_train1_given_cls(i,1)*p_data_train1_given_cls(i,2)) /3 ;
    ans_train1(i,2)=   (p_data_train1_given_cls(i,3)*p_data_train1_given_cls(i,4)) /3 ;
    ans_train1(i,3)=   (p_data_train1_given_cls(i,5)*p_data_train1_given_cls(i,6)) /3 ;
end

calculated_label_train_c1 = zeros(100,1);
confusion_mat_train =  zeros(3,3);
for i = 1 :250
    label_count = max(max(ans_train1(i,1),ans_train1(i,2)),ans_train1(i,3));
    if (ans_train1(i,1) == ans_train1(i,2) )
        calculated_label_train_c1(i,1) = 1;
    elseif label_count == ans_train1(i,1)
       calculated_label_train_c1(i,1) = 1;
       confusion_mat_train(1,1) = confusion_mat_train(1,1)+1;
    elseif label_count == ans_train1(i,2)
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
count = 0 ;
p_data_train2_given_cls = zeros(250,6);
for i = 1:250
    p_data_train2_given_cls(i,1) = exp(-((power(AM2(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train2_given_cls(i,2) = exp(-((power(AM2(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train2_given_cls(i,3) = exp(-((power(AM2(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train2_given_cls(i,4) = exp(-((power(AM2(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train2_given_cls(i,5) = exp(-((power(AM2(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train2_given_cls(i,6) = exp(-((power(AM2(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_train2 = zeros(250,3);

for i= 1:250
    ans_train2(i,1) =  (p_data_train2_given_cls(i,1)*p_data_train2_given_cls(i,2)) /3 ;
    ans_train2(i,2)=   (p_data_train2_given_cls(i,3)*p_data_train2_given_cls(i,4)) /3 ;
    ans_train2(i,3)=   (p_data_train2_given_cls(i,5)*p_data_train2_given_cls(i,6)) /3 ;
end

calculated_label_train_c2 = zeros(250,1) ;
for i = 1 :250
    label_count = max(max(ans_train2(i,1),ans_train2(i,2)),ans_train2(i,3));
    if (ans_train2(i,1) == ans_train2(i,2) )
        calculated_label_train_c2(i,1) = 1;
    elseif label_count == ans_train2(i,1)
       calculated_label_train_c2(i,1) = 1;
       confusion_mat_train(2,1) = confusion_mat_train(2,1)+1;
    elseif label_count == ans_train2(i,2)
        calculated_label_train_c2(i,1) = 2;
        confusion_mat_train(2,2) = confusion_mat_train(2,2)+1;
    else
        calculated_label_train_c2(i,1) = 3;
        confusion_mat_train(2,3) = confusion_mat_train(2,3)+1;
    end
     if(calculated_label_train_c2(i,1) ~= 1)
         count = count+1;
     end 
end

count = 0 ;
p_data_train3_given_cls = zeros(250,6);
for i = 1:250
    p_data_train3_given_cls(i,1) = exp(-((power(AM3(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train3_given_cls(i,2) = exp(-((power(AM3(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train3_given_cls(i,3) = exp(-((power(AM3(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train3_given_cls(i,4) = exp(-((power(AM3(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_data_train3_given_cls(i,5) = exp(-((power(AM3(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_data_train3_given_cls(i,6) = exp(-((power(AM3(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_train3 = zeros(250,3);

for i= 1:250
    ans_train3(i,1) =  (p_data_train3_given_cls(i,1)*p_data_train3_given_cls(i,2)) /3 ;
    ans_train3(i,2)=   (p_data_train3_given_cls(i,3)*p_data_train3_given_cls(i,4)) /3 ;
    ans_train3(i,3)=   (p_data_train3_given_cls(i,5)*p_data_train3_given_cls(i,6)) /3 ;
end

calculated_label_train_c3 = zeros(250,1) ;
for i = 1 :250
    label_count = max(max(ans_train3(i,1),ans_train3(i,2)),ans_train3(i,3));
    if (ans_train3(i,1) == ans_train3(i,2) )
        calculated_label_train_c3(i,1) = 1;
    elseif label_count == ans_train3(i,1)
       calculated_label_train_c3(i,1) = 1;
       confusion_mat_train(3,1) = confusion_mat_train(3,1)+1;
    elseif label_count == ans_train3(i,2)
        calculated_label_train_c3(i,1) = 2;
        confusion_mat_train(3,2) = confusion_mat_train(3,2)+1;
    else
        calculated_label_train_c3(i,1) = 3;
        confusion_mat_train(3,3) = confusion_mat_train(3,3)+1;
    end
     if(calculated_label_train_c3(i,1) ~= 1)
         count = count+1;
     end 
end

accuracy_train =( confusion_mat_train(1,1)+confusion_mat_train(2,2)+confusion_mat_train(3,3))/750  ;
accuracy_test = (confusion_mat_test(1,1)+confusion_mat_test(2,2)+confusion_mat_test(3,3))/300  ;
accuracy_valid= (confusion_mat_val(1,1)+confusion_mat_val(2,2)+confusion_mat_val(3,3))/450 ;


%%%%%%%%%%%%%%%%plot
%%%%%%%%%%%%%%%%plot

xaxis = -10:0.05:20;
yaxis = -20:0.05:20;
[x, y] = meshgrid(xaxis, yaxis);
plot_data = [x(:) y(:)];
total_plot_data = size(plot_data,1);
predicted_plot = zeros(total_plot_data,1);


count = 0 ;
p_plot_data = zeros(total_plot_data,6);

for i = 1:total_plot_data
    p_plot_data(i,1) = exp(-((power(plot_data(i,1)-mean1(1,1) , 2))/(2*sigma_square)))   ;
    p_plot_data(i,2) = exp(-((power(plot_data(i,2)-mean1(2,1) , 2))/(2*sigma_square)))   ;
    
    p_plot_data(i,3) = exp(-((power(plot_data(i,1)-mean2(1,1) , 2))/(2*sigma_square)))   ;
    p_plot_data(i,4) = exp(-((power(plot_data(i,2)-mean2(2,1) , 2))/(2*sigma_square)))   ;
    
    p_plot_data(i,5) = exp(-((power(plot_data(i,1)-mean3(1,1) , 2))/(2*sigma_square)))   ;
    p_plot_data(i,6) = exp(-((power(plot_data(i,2)-mean3(2,1) , 2))/(2*sigma_square)))   ;
end

ans_plot_data = zeros(total_plot_data,3);

for i= 1:total_plot_data
    ans_plot_data(i,1) =  (p_plot_data(i,1)*p_plot_data(i,2)) /3 ;
    ans_plot_data(i,2)=   (p_plot_data(i,3)*p_plot_data(i,4)) /3 ;
    ans_plot_data(i,3)=   (p_plot_data(i,5)*p_plot_data(i,6)) /3 ;
    
    label_count = max(max(ans_plot_data(i,1),ans_plot_data(i,2)),ans_plot_data(i,3));
    if (ans_plot_data(i,1) == ans_plot_data(i,2) )
        predicted_plot(i,1) = 1;
    elseif label_count == ans_plot_data(i,1)
       predicted_plot(i,1) = 1;
    %   confusion_mat_test(3,1) = confusion_mat_test(3,1)+1;
    elseif label_count == ans_plot_data(i,2)
        predicted_plot(i,1) = 2;
      %  confusion_mat_test(3,2) = confusion_mat_test(3,2)+1;
    else
        predicted_plot(i,1) = 3;
       % confusion_mat_test(3,3) = confusion_mat_test(3,3)+1;
    end
end



plot = reshape(predicted_plot, size(x));
imagesc(xaxis,yaxis,plot);
set(gca,'ydir','normal');

hold on;
scatter(AM1(:,1),AM1(:,2),'m','x');
scatter(AM2(:,1),AM2(:,2),'w','x');
scatter(AM3(:,1),AM3(:,2),'r','x');

xlabel('x1')
ylabel('x2')
str  = { strcat('Decision Region Plot')};
title(str,'FontSize',15);

t = text(-5,10,'CLASS 1','Color','r','FontSize',14);
t1 = text(5,15,'CLASS 2','Color','b','FontSize',14);
t3 = text(5,-10,'CLASS 3','Color','b','FontSize',14);
hold off;


