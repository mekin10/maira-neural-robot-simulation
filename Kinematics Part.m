 %% Forward Kinematics for MAiRA Neura Robot
% DH Parameters for MAiRA Neura Robot (7-DOF, in meters)
% Using standard (classic) Denavit-Hartenberg convention
% All joints are revolute

L1 = Link('d', 0.429, 'a', 0,     'alpha', -pi/2);
L2 = Link('d', 0,     'a', 0,     'alpha',  pi/2);
L3 = Link('d', 0.406, 'a', 0.106, 'alpha',  pi/2);
L4 = Link('d', 0,     'a', -0.106,'alpha', -pi/2);
L5 = Link('d', 0.494, 'a', 0,     'alpha',  pi/2);
L6 = Link('d', 0,     'a', 0.113, 'alpha',  pi/2);
L7 = Link('d', 0.138, 'a', 0,     'alpha',  0);

% Create the SerialLink robot object (7-DOF)
maira = SerialLink([L1 L2 L3 L4 L5 L6 L7], 'name', 'MAiRA Neura');

% Display the robot model for verification
maira

% Define a sample home configuration (all zeros)
q_home = zeros(1,7);

% Forward Kinematics at home position
T_home = maira.fkine(q_home);

disp('-----------------------------------------------');
disp('Home Pose T_home (4x4 Homogeneous Matrix):');
disp(T_home.T);  % .T extracts the 4x4 matrix if using SE3
disp('-----------------------------------------------');

% Plot the robot at home position
figure(1);
maira.plot(q_home);
title('MAiRA Neura at Home Configuration');

%% Inverse Kinematics (Numerical method for redundant 7-DOF)
%--- 1. Define Target Pose (T_target) ---
% Choose a known reachable joint configuration to create a valid target pose

q_target_deg = [0, 30, 0, 60, 0, 0, 45];  % Example reachable configuration (degrees)
q_target_rad = q_target_deg * (pi/180);

T_target = maira.fkine(q_target_rad).T;  % Extract 4x4 matrix

disp('-----------------------------------------------');
disp('Target Pose T_target (4x4 Homogeneous Matrix):');
disp(T_target);
disp('-----------------------------------------------');

%--- 2. Numerical IK Solver (Damped Pseudoinverse with optional joint centering) ---
q_guess = zeros(1,7);               % Initial guess (row vector)
q = q_guess;

tol = 1e-8;                         % Convergence tolerance
max_iter = 1000;
lambda = 0.01;                      % Damping factor for stability near singularities
alpha = 0.5;                        % Step size (can be 1.0 if convergence is good)
k_n = 0.05;                         % Gain for secondary task (joint centering)

q_desired = zeros(1,7);             % Desired joint positions for centering (home)

for iter = 1:max_iter
    % Current pose
    T_current = maira.fkine(q).T;
    p_current = T_current(1:3,4);
    R_current = T_current(1:3,1:3);
    
    % Target
    p_target = T_target(1:3,4);
    R_target = T_target(1:3,1:3);
    
    % Position error
    Delta_p = p_target - p_current;
    
    % Orientation error (approximate axis-angle vector for small errors)
    err_R = R_target.' * R_current;
    rx = err_R(:,1); ry = err_R(:,2); rz = err_R(:,3);
    Delta_omega = 0.5 * (cross(rx, ry) + cross(ry, rz) + cross(rz, rx));
    
    % Total task error
    e = [Delta_p; Delta_omega];
    
    if norm(e) < tol
        fprintf('Converged in %d iterations\n', iter);
        break;
    end
    
    % Jacobian in base frame (6x7)
    J = maira.jacob0(q);
    
    % Damped pseudoinverse
    Jt = J.';
    JJt = J * Jt;
    Jp = Jt * inv(JJt + lambda^2 * eye(6));
    
    % Primary task
    Delta_q_primary = Jp * e;       % 7x1
    
    % Secondary task: joint centering (null-space projection)
    q_error = (q - q_desired).';    % 7x1
    Delta_q_secondary = -k_n * q_error;
    null_proj = eye(7) - Jp * J;
    Delta_q_secondary = null_proj * Delta_q_secondary;
    
    % Total joint increment
    Delta_q = Delta_q_primary + Delta_q_secondary;  % 7x1
    
    % Update
    q = q + alpha * Delta_q.';
end

%--- 3. Display Results ---
q_solution_deg = q * (180/pi);

disp('Inverse Kinematics Solution q (Joint Angles in degrees):');
fprintf('[J1, J2, J3, J4, J5, J6, J7]\n');
fprintf('%.4f   ', q_solution_deg);
fprintf('\n');

% Verification
T_verification = maira.fkine(q).T;
disp('-----------------------------------------------');
disp('FK Verification (Pose achieved by IK solution):');
disp(T_verification);


% Pose error
error = norm(T_target - T_verification, 'fro');
fprintf('\nPose Error Magnitude (Frobenius norm, should be near zero): %.6e\n', error);

% Plot the solution
figure(2);
maira.plot(q);
title('MAiRA Neura at Inverse Kinematics Solution');

%% Jacobian
%--- 1. Define a Test Configuration ---
q_test_deg = [0, 30, 0, 60, 0, 0, 45];
q_test_rad = q_test_deg * (pi/180);

%--- 2. Calculate Jacobians ---
% Jacobian in base frame (6x7 for 7-DOF robot)
J_BaseFrame = maira.jacob0(q_test_rad);

disp('----------------------------------------------------');
disp('J_BaseFrame (J_0): 6x7 matrix (End-effector twist in BASE frame)');
disp('Rows 1-3: Linear velocity, Rows 4-6: Angular velocity');
disp(J_BaseFrame);

% Jacobian in end-effector frame (6x7)
J_EndEffectorFrame = maira.jacobe(q_test_rad);

disp('----------------------------------------------------');
disp('J_EndEffectorFrame (J_E): 6x7 matrix (End-effector twist in EE frame)');
disp(J_EndEffectorFrame);

% Plot at test configuration
figure(3);
maira.plot(q_test_rad);
title('MAiRA Neura at Jacobian Test Configuration');

%% MAiRA Neura Kinematic Trajectory Generation (Joint Space)
%--- 1. Define Start and End Configurations ---
q_start_rad = zeros(1,7);  % Home position

q_end_deg = [30, 20, 40, -30, 10, 20, 45];  % Example end configuration
q_end_rad = q_end_deg * (pi/180);

%--- 2. Trajectory Parameters ---
T_f = 5;                     % Duration (seconds)
steps = 100;                 % Number of points
t = linspace(0, T_f, steps); % Time vector

%--- 3. Generate Joint Space Trajectory (quintic polynomial via jtraj) ---
[Q_rad, QD_rad, QDD_rad] = jtraj(q_start_rad, q_end_rad, t);

% Convert position to degrees for display
Q_deg = Q_rad * (180/pi);

%--- 4. Visualization ---
% Animate the trajectory
figure(4);
maira.plot(Q_rad);
title('MAiRA Neura Joint-Space Trajectory');

% Plot kinematic profiles for Joint 1 (example)
figure(5);
subplot(3,1,1);
plot(t, Q_deg(:,1));
ylabel('Position (deg)');
title('Joint 1 Profile');

subplot(3,1,2);
plot(t, QD_rad(:,1)*180/pi);
ylabel('Velocity (deg/s)');

subplot(3,1,3);
plot(t, QDD_rad(:,1)*180/pi);
ylabel('Acceleration (deg/s^2)');
xlabel('Time (s)');
