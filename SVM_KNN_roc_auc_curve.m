clc;
clear all;
close all;

Train_Feat = coder.load('Train_Feat.mat');
Train_Label = coder.load('Train_Label.mat'); 
Test_Feat = coder.load('Test_Feat.mat');
Test_Label = coder.load('Test_Label.mat');
Train_Feat=struct2cell(Train_Feat);
Train_Label=struct2cell(Train_Label);
Test_Feat=struct2cell(Test_Feat);
Test_Label=struct2cell(Test_Label);
 
train_Feat=Train_Feat{9,1};
train_Label=Train_Label{10,1};
test_Feat=Test_Feat{4,1};
test_Label=Test_Label{5,1};
t_f= train_Feat(1:5504,:);
t_l=train_Label(1:5504,:);
knn = ClassificationKNN.fit(t_f,t_l, 'NumNeighbors',5,'Standardize',1);
[~,score_knn]  = resubPredict(knn);
[Xknn1,Yknn1,Tknn1,AUCknn1] = perfcurve(test_Label,score_knn(:,1),'1');


Y=t_l;
resp1 = ismember(Y,1);
mdlSVM1 = fitcsvm(t_f,t_l,'Standardize',true);
mdlSVM1 = fitPosterior(mdlSVM1);
[~,score_svm1] = resubPredict(mdlSVM1);
score_svm1(5504,2)=0;
[Xsvm1,Ysvm1,Tsvm1,AUCsvm1] = perfcurve(test_Label,score_svm1(:,2),'true');

figure;
plot(Xsvm1,Ysvm1,'LineWidth', 2);
hold on
plot(Xknn1,Yknn1, 'LineWidth', 2);

xlabel('False Positive Ratio (1-specificity)','fontsize',10,'FontWeight','bold');
ylabel('True Positive Ratio (Sensitivity)','fontsize',10,'FontWeight','bold');
title('ROC For Dermatitis VS All') 
grid on
ROCtitle_1=['Dermatitis AUC-SVM = ',num2str(roundn(AUCsvm1,-3))]; 
ROCtitle_2=['Dermatitis AUC-KNN = ',num2str(roundn(AUCknn1,-3))];
hh1=legend((ROCtitle_1),(ROCtitle_2),'Location','southeast');
set(hh1,'edgecolor','black')
%%%%%%

[Xknn2,Yknn2,Tknn2,AUCknn2] = perfcurve(test_Label,score_knn(:,2),'2');

resp2 = ismember(Y,2);
mdlSVM2 = fitcsvm(test_Feat,resp2,'Standardize',true);
mdlSVM2 = fitPosterior(mdlSVM2);
[~,score_svm2] = resubPredict(mdlSVM2);
score_svm2(5504,2)=0;
[Xsvm2,Ysvm2,Tsvm2,AUCsvm2] = perfcurve(resp2,score_svm2(:,2),'true');

figure;
plot(Xsvm2,Ysvm2,'LineWidth', 2);
hold on
plot(Xknn2,Yknn2, 'LineWidth', 2);

xlabel('False Positive Ratio (1-specificity)','fontsize',10,'FontWeight','bold');
ylabel('True Positive Ratio (Sensitivity)','fontsize',10,'FontWeight','bold');
title('ROC For Impetigo VS All') 
grid on
ROCtitle_1=['Impetigo AUC-SVM = ',num2str(roundn(AUCsvm2,-3))];
ROCtitle_2=['Impetigo AUC-KNN = ',num2str(roundn(AUCknn2,-3))];
hh1=legend((ROCtitle_1),(ROCtitle_2),'Location','southeast');
set(hh1,'edgecolor','black')

%%%%%%%%%%%%

[Xknn3,Yknn3,Tknn3,AUCknn3] = perfcurve(test_Label,score_knn(:,3),'3');
resp3 = ismember(Y,3);
mdlSVM3 = fitcsvm(test_Feat,resp3,'Standardize',true);
mdlSVM3 = fitPosterior(mdlSVM3);
[~,score_svm3] = resubPredict(mdlSVM3);
score_svm3(5504,2)=0;
[Xsvm3,Ysvm3,Tsvm3,AUCsvm3] = perfcurve(resp3,score_svm3(:,2),'true');


figure;
plot(Xsvm3,Ysvm3,'LineWidth', 2);
hold on
plot(Xknn3,Yknn3, 'LineWidth', 2);

xlabel('False Positive Ratio (1-specificity)','fontsize',10,'FontWeight','bold');
ylabel('True Positive Ratio (Sensitivity)','fontsize',10,'FontWeight','bold');
title('ROC For Melanoma VS All')
grid on
ROCtitle_1=['Melanoma AUC-SVM = ',num2str(roundn(AUCsvm3,-3))];
ROCtitle_2=['Melanoma AUC-KNN = ',num2str(roundn(AUCknn3,-3))];
hh1=legend((ROCtitle_1),(ROCtitle_2),'Location','southeast');
set(hh1,'edgecolor','black')

%%%%%%%%%%%%

[Xknn4,Yknn4,Tknn4,AUCknn4] = perfcurve(test_Label,score_knn(:,4),'4');
resp4 = ismember(Y,4);
mdlSVM4 = fitcsvm(test_Feat,resp4,'Standardize',true);
mdlSVM4 = fitPosterior(mdlSVM4);
[~,score_svm4] = resubPredict(mdlSVM4);
score_svm4(5504,2)=0;
[Xsvm4,Ysvm4,Tsvm4,AUCsvm4] = perfcurve(resp4,score_svm4(:,2),'true');

figure;
plot(Xsvm4,Ysvm4,'LineWidth', 2);
hold on
plot(Xknn4,Yknn4, 'LineWidth', 2);

xlabel('False Positive Ratio (1-specificity)','fontsize',10,'FontWeight','bold');
ylabel('True Positive Ratio (Sensitivity)','fontsize',10,'FontWeight','bold');
title('ROC For Diabetic Foot Ulcer VS All')
grid on
ROCtitle_1=['Diabetic Foot Ulcer AUC-SVM = ',num2str(roundn(AUCsvm4,-3))];
ROCtitle_2=['Diabetic Foot Ulcer AUC-KNN = ',num2str(roundn(AUCknn4,-3))];
hh1=legend((ROCtitle_1),(ROCtitle_2),'Location','southeast');
set(hh1,'edgecolor','black')


%%%%%%%%%%%%

