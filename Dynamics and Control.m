%% ==============================================================
%  MAiRA Neura Robot – Full Kinematics + Dynamics + Control
%  Using Robotics Toolbox (Peter Corke)
%% ==============================================================

clc; clear; close all;

deg = pi/180;   % Degree to radian

%% ==============================================================
%% 1. LINK PARAMETERS (YOUR PROVIDED DATA)
%% ==============================================================

d1 = 0.429;
d2 = 0;
d3 = 0.406;
d4 = 0;
d5 = 0.494;
d6 = 0;

% -------- Link 1 --------
L(1) = Revolute('d', d1,'a',0,'alpha',-pi/2,'offset',0,...
    'I',[0.0029525 0.0060091 0.0058821 0 0 0],...
    'r',[d1/2 0 -0.03],'m',1.5,...
    'Jm',13.5e-6,'G',1/156,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-360 360]*deg,'standard');

% -------- Link 2 --------
L(2) = Revolute('d',d2,'a',0,'alpha',pi/2,'offset',0,...
    'I',[0.0029525 0.0060091 0.0058821 0 0 0],...
    'r',[d1/2 0 0],'m',1.5,...
    'Jm',13.5e-6,'G',1/156,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-180 180]*deg,'standard');

% -------- Link 3 --------
L(3) = Revolute('d',d2,'a',0.106,'alpha',pi/2,'offset',0,...
    'I',[0.00172767 0.00041967 0.0018468 0 0 0],...
    'r',[d1/2 0 0],'m',1.6,...
    'Jm',13.5e-6,'G',1/100,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-360 360]*deg,'standard');

% -------- Link 4 --------
L(4) = Revolute('d',d3,'a',-0.106,'alpha',-pi/2,...
    'I',[0.0006764 0.0010573 0.0006610 0 0 0],...
    'r',[0 d1/2 0],'m',1.8,...
    'Jm',9.25e-6,'G',1/71,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-180 180]*deg,'standard');

% -------- Link 5 --------
L(5) = Revolute('d',d5,'a',0,'alpha',pi/2,'offset',0,...
    'I',[0.0001934 0.0001602 0.0000689 0 0 0],...
    'r',[0 0 0],'m',1,...
    'Jm',3.5e-6,'G',1/71,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-360 360]*deg,'standard');

% -------- Link 6 --------
L(6) = Revolute('d',d6,'a',0.113,'alpha',pi/2,...
    'I',[0.0001934 0.0001602 0.0000689 0 0 0],...
    'r',[0 0 0],'m',0.5,...
    'Jm',3.5e-6,'G',1/71,'B',1.48e-3,...
    'Tc',[0.395 -0.435],'qlim',[-90 90]*deg,'standard');

% -------- Link 7 (end-effector, light payload assumption) --------
L(7) = Revolute('d',0.138,'a',0,'alpha',0,...
    'I',[1e-4 1e-4 1e-4 0 0 0],...
    'r',[0 0 0.05],'m',0.3,...
    'qlim',[-360 360]*deg,'standard');

%% ==============================================================
%% 2. CREATE MAiRA ROBOT
%% ==============================================================

maira = SerialLink(L, 'name', 'MAiRA Neura');

disp('MAiRA Robot Model with Full Dynamics:');
maira

figure(1);
maira.plot(zeros(1,7));
title('MAiRA Neura – Home Configuration');

%% ==============================================================
%% 3. TRAJECTORY GENERATION
%% ==============================================================

q_start = zeros(1,7);
q_end   = [30 20 40 -30 10 20 45]*deg;

Tf = 5;
t = linspace(0,Tf,100);

[q, qd, qdd] = jtraj(q_start, q_end, t);

%% ==============================================================
%% 4. COMPUTED TORQUE CONTROL (PD + DYNAMICS)
%% ==============================================================

Kp = diag([120 120 120 100 80 60 40]);
Kd = diag([25 25 25 20 15 10 8]);

q_act  = q_start;
qd_act = zeros(1,7);

tau_log = zeros(length(t),7);

dt = t(2)-t(1);

for k = 1:length(t)

    e  = q(k,:)  - q_act;
    ed = qd(k,:) - qd_act;

    M = maira.inertia(q_act);
    C = maira.coriolis(q_act, qd_act);
    G = maira.gravload(q_act)';

    tau = M*(qdd(k,:)' + Kd*ed' + Kp*e') + C*qd_act' + G;

    qdd_act = M \ (tau - C*qd_act' - G);

    qd_act = qd_act + qdd_act'*dt;
    q_act  = q_act  + qd_act*dt;

    tau_log(k,:) = tau';
end

figure(2);
plot(t, tau_log);
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Computed Torque Control – Joint Torques');
grid on;

%% ==============================================================
%% 5. ENERGY & POWER ANALYSIS
%% ==============================================================

power = sum(abs(tau_log .* qd),2);
energy = cumtrapz(t, power);

figure(3);
subplot(2,1,1);
plot(t, power,'LineWidth',1.5);
ylabel('Power (W)');
title('Instantaneous Power');

subplot(2,1,2);
plot(t, energy,'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Energy (J)');
title('Total Energy Consumption');

%% ==============================================================
%% 6. FINAL TRAJECTORY ANIMATION
%% ==============================================================

figure(4);
maira.plot(q);
title('MAiRA Neura – Controlled Motion');

