P0 = [1, 3, 0]; 
P1 = [7, 2, 0]; 
T0 = [9, 5, 0]; 
T1 = [-1, -2, 0]; 

h00 = @(t) (2*t.^3 - 3*t.^2 + 1);
h10 = @(t) (t.^3 - 2*t.^2 + t);
h01 = @(t) (-2*t.^3 + 3*t.^2);
h11 = @(t) (t.^3 - t.^2);

t = linspace(0, 1, 100);
H = (h00(t)' * P0 + h10(t)' * T0 + h01(t)' * P1 + h11(t)' * T1);

figure;
plot3(H(:,1), H(:,2), H(:,3), 'b-', 'LineWidth', 2);
hold on;
plot3(P0(1), P0(2), P0(3), 'ro', 'MarkerSize', 10);
plot3(P1(1), P1(2), P1(3), 'go', 'MarkerSize', 10);
quiver3(P0(1), P0(2), P0(3), T0(1), T0(2), T0(3), 0.5, 'r', 'LineWidth', 1.5);
quiver3(P1(1), P1(2), P1(3), T1(1), T1(2), T1(3), 0.5, 'g', 'LineWidth', 1.5);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Hermite Curve');
grid on;
axis equal;
hold off;
