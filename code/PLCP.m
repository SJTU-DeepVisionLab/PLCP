function [acbase, accuracy_testcom] = PLCP(train_data,train_p_target,test_data, test_target,par,parameter, ga_q, kp, gamma, mutual)

% parameters for PL-AGGD
ker  = 'rbf'; 
k = 10;
lambda = 1;
mu = 1;
gama = 0.05;
Maxiter = 10;

% labeling confidence
ppp = build_label_manifold(train_data,train_p_target,k);

% non-candidate labeling confidence
ppf=1-train_p_target;

y=ppp;


[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
for j = 1:mutual
    % PL-AGGD
    for i = 1:Maxiter
        W = obtain_W(train_data,y,k,lambda,mu);
        y = UpdateY(W,train_p_target,train_outputs,mu);
        [train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
    end

    if j == 1
        accuracy_test = CalAccuracy(test_outputs, test_target);
        fprintf('The accuracy of PL-AGGD is: %f \n',accuracy_test);
        acbase = accuracy_test;
    end
    
    ppp = parameter* ppp + (1-parameter) * train_outputs;
    ppp = min(train_p_target, max(0, ppp));
    
    tep = exp(kp*ppp);
    tep = tep .* train_p_target;
    tep = full(tep);
    [row,~] = size(tep);
    
    for iter=1:row
        tep(iter,:) = tep(iter,:) / sum(tep(iter,: ));
    end
   
    q = 1-tep;

    for i = 1:30
        [train_outputscom, test_outputscom]=MulRegression(train_data, q, test_data, gamma, par, ker);
        q = Update_Q_com(tep, train_outputscom, ga_q, 1-train_p_target);
    end
    
    accuracy_testcom = CalAccuracy(1-test_outputscom, test_target);
    fprintf('The accuracy of PL-AGGD-PLCP is: %f \n',accuracy_testcom);
    
    ppf = parameter * ppf + (1-parameter) * train_outputscom;
    ppf = min(1, max(1-train_p_target, ppf));
    
    tep = 1 - ppf;
    tep = exp(kp*tep);
    tep = tep .* train_p_target;
    
    [row,~] = size(tep);
    for iter=1:row
        tep(iter,:) = tep(iter,:) / sum(tep(iter,:));
    end
    
    y = min(train_p_target, tep);
end


end