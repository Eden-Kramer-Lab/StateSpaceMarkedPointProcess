%% Load behavioral data and prepare it for the toolbox
clear
close all
clc

load('StateVar.mat');
% put 1, 2, or 3
Data    = StateVar(1).Data;

%% Data Preparation
inv_dt  = 50;         % 20 msec update time
K0      = round(Data(end,3)*inv_dt)+2;
x_ind   = find(isfinite(Data(:,3)));
Yn      = zeros(K0,1); Yn(round(Data(x_ind,3)*inv_dt)) = 1;
Yb      = zeros(K0,1); Yb(round(Data(x_ind,3)*inv_dt)) = Data(x_ind,4);

range_t = 1.0;
Data_x  = zeros(K0,1);
Data_x(round(Data(x_ind,2)*inv_dt),1) = Data(x_ind,1);
obs_valid = zeros(K0,1);
In        = zeros(K0,4);
temp_a    = zeros(K0,1);
temp_b    = zeros(K0,1);
temp_a(round(Data(:,2)*inv_dt)) = 1;
temp_b(round(Data(:,3)*inv_dt)) = 1;
obs_state = 0;
incong    = 0;
count     = 0;
count_state = 0;
SPK = [];
for k=1:K0
    obs_valid(k)    = obs_state;
    % 1 on task interval
    In(k,1)         = obs_state;
    % 1 on incongruent interval
    In(k,2)         = incong;
    % event kernel, linearly increases by time (zero to positive)
    In(k,3)         = obs_state   * log(1+exp((count-0.30*inv_dt)));
    % onset kernel, linearly increase from onset (negative to zero)
    In(k,4)         = - obs_state * log(1+exp((0.5*inv_dt-count)));
    
    % count obs_valid
    if temp_a(k) == 1 && obs_state ==0
        obs_state = 1;
        if Data_x(k,1) == 2
            incong = 1;
        end
        count = 0;
    end
    if temp_b(k)==1  && obs_state ==1
        obs_state = 0;
        incong    = 0;
        count     = 0;
    end
    if obs_state == 1
        count = count + 1;
    end
end


%%  Model Setting
Iter = 20;
Uk   = zeros(K0,1);
Ib   = In;
% create model
Param = compass_create_state_space(2,1,4,4,eye(2,2),[1 2],[0 0],[1 2],[1 1]);
Param.dt  = 1/inv_dt;
Param.ws  = 0.1;
% set learning parameters
Param = compass_set_learning_param(Param,Iter,0,1,1,1,1,1,1,2,0);
% initialize model parameters
Param.Ck  = [1 1];
Param.Dk  = [0 0 1 1];
Param.Ek  = [1 1];
Param.Fk  = [0 0 1 1];
DISTR     = [3 1];% Point Process and Bernoulli observation
Param_ini = [Param.W0(1,1),Param.W0(2,2),Param.Bk(1,1),Param.Bk(2,1),Param.Ek(1,1),Param.Ek(1,2),Param.Fk(1,3),Param.Fk(1,4)]';
[XSmt,SSmt,Param,XPos,SPos,ML,YP,YB] = compass_em_e(DISTR,Uk,In,Ib,Yn,Yb,Param,obs_valid);
Param_est = [Param.W0(1,1),Param.W0(2,2),Param.Bk(1,1),Param.Bk(2,1),Param.Ek(1,1),Param.Ek(1,2),Param.Fk(1,3),Param.Fk(1,4)]';

% Generate trajectory
Ns = 10;
Xs = compass_state_sample(10,XSmt,SSmt,XPos,SPos,Param,Uk);

%% Figures
figure()% Conggruent and Incongruent trials
ind = find(Data(:,1)==1);
plot(Data(ind,2),Data(ind,3)-Data(ind,2),'*','LineWidth',2);hold on
ind = find(Data(:,1)==2);
plot(Data(ind,2),Data(ind,3)-Data(ind,2),'o','LineWidth',2);hold on
ind = find(Data(:,4)==0);
plot(Data(ind,2),Data(ind,3)-Data(ind,2),'g+','LineWidth',2);
legend('Conggruent Trial','Inconggruent Trial','Orientation','horizontal','Location','northoutside')
hold off
ylabel('Reaction time and Congruent/Incongruent Decision')
xlabel('Trials` Onset Time (Sec)')
set(gca, 'FontSize', 16)
axis tight

% Estimated parameres:
% % figure()
% % % subplot(3,3,[4 5 7 8])
% % Mat = [Param_ini Param_est];
% % bar(Mat)
% % axis tight
% % ylabel('\theta')
% % set(gca,'xTickLabel',{'\sigma_b^2','\sigma_i^2','c_1','c_2','d_1','d_2','d_3','d_4'});
% % legend('Initial values','Estimated values','Location','southwest')
% % % ax = gca;
% % % area = [0.5 -0.01 2.5 0.01];
% % % inlarge = subplot(3,3,3);
% % % panpos = inlarge.Position;
% % % delete(inlarge);
% % % inlarge = zoomin(ax,area,panpos);


xm   = zeros(K0,1);
xb   = zeros(K0,1);
yy_m = zeros(K0,1);
% x_b:
for i=1:K0
    temp=XSmt{i};a_xm= temp(1);
    temp=SSmt{i};a_xb= temp(1,1);
    
    xm(i) = a_xm;
    xb(i) = a_xb;
    
    yy_m(i) = exp(xm(i) + 0.5*xb(i)) ;
    
end
figure()
compass_plot_bound(1,(1:K0)*Param.dt,xm,(xm-2*sqrt(xb))',(xm+2*sqrt(xb))');
ylabel('x_b');
axis tight
xlabel('Time (Sec)')
hold off

% x_i:
xm = zeros(K0,1);
xb = zeros(K0,1);
xx_m = zeros(K0,1);
for i=1:K0
    temp=XSmt{i};a_xm= temp(2);
    temp=SSmt{i};a_xb= temp(2,2);
    
    xm(i) = a_xm;
    xb(i) = a_xb;
    
    xx_m(i) = exp(sum(XSmt{i}) + 0.5*[1 1]*SSmt{i}*[1 1]') ;
end
figure()
compass_plot_bound(1,(1:K0)*Param.dt,xm,(xm-2*sqrt(xb))',(xm+2*sqrt(xb))');
ylabel('x_i');
axis tight
xlabel('Time (Sec)')
hold off

figure()
plot((1:length(Yn))*Param.dt,xx_m,'LineWidth',2);hold on;
plot((1:length(Yn))*Param.dt,yy_m,'LineWidth',2);hold off;
ylabel('Expected Rate of Response');
axis tight
xlabel('Time (Sec)')
hold off









