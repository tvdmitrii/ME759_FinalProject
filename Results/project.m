cpu = xlsread("res_cpu.xlsx");
gpu_new = xlsread("res_gpu_new.xlsx");
gpu_old = xlsread("res_gpu_old.xlsx");
gpu_new_1060 = xlsread("res_gpu_new_1060.xlsx");
gpu_old_1060 = xlsread("res_gpu_old_1060.xlsx");

N = cpu(:,1);
time_cpu = cpu(:,3);
time_gpu_new = gpu_new(:,3);
time_gpu_old = gpu_old(:,3);
time_gpu_new_1060 = gpu_new_1060(:,3);
time_gpu_old_1060 = gpu_old_1060(:,3);

figure;
semilogy(N, time_cpu,'-o');
hold on; grid on;
semilogy(N,time_gpu_new,'-o');
legend("CPU","GPU", 'Location','southeast');
xlabel("Number of Points N");
ylabel("Time [s]");
title("GPU vs CPU Compute Time Comparison. m = 8");
print('-painters','-dpng','-r300',"cpu_gpu");

figure;
plot(N, time_gpu_old_1060,'-o');
hold on; grid on;
plot(N,time_gpu_new_1060,'-o');
legend("GPU, No Streams","GPU, Streams", 'Location','southeast');
xlabel("Number of Points N");
ylabel("Time [s]");
title("Compute Time Comparison. GTX 1060 3GB. m = 8");
print('-painters','-dpng','-r300',"gpu_1060");

figure;
plot(N, time_gpu_old,'-o');
hold on; grid on;
plot(N,time_gpu_new,'-o');
legend("GPU, No Streams","GPU, Streams", 'Location','southeast');
xlabel("Number of Points N");
ylabel("Time [s]");
title("Compute Time Comparison. GTX TITAN X 12GB. m = 8");
print('-painters','-dpng','-r300',"gpu_titan");
