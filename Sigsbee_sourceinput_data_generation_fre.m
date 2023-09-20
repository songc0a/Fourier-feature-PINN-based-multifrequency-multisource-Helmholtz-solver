clear all 
close all

load('bluewhitered.mat')

str2 =['vpt'];
filename2=['' str2 '.rsf@'];
fid2=fopen(filename2,'rb');
vel=fread(fid2,[143,465],'float');
fclose(fid2);

v=vel(1:1:101,1:1:101) ;%v
% v=vel(1:1:101,101:1:201) ;%v

v= imgaussfilt(v,1);

v = v/1000; 

n = size(v);
dx = 0.025; dz = 0.025;
nx = n(2); nz = n(1);
h  = [dz dx];
x = (0:nx-1)*dx;
z = (0:nz-1)*dz;

figure;
pcolor(x,z,v);
shading interp
axis ij
xlabel('X (km)','FontSize',12)
ylabel('Z (km)','FontSize',12);
colormap( (bluewhitered))
colorbar
caxis([1.5 3])
set(gca,'FontSize',14)

N_train = 50000; %% number of the random points

z_train  =  rand(N_train,1)*2.45 + 0.05;
x_train  =  rand(N_train,1)*2.45 + 0.05;
f_train  = 5*rand(N_train,1)+5.0;
sx_train =  rand(N_train,1)*2.45 + 0.05;

v0 = ones(N_train,1)*1.830;

src_z = 2; 
sz = (src_z-1)*dz; %% depth of the source

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);
[X,Y] = meshgrid(x,z);
x1 = [0:2501-1]*0.001;
z1 = [0:2501-1]'*0.001;
[Xq,Yq] = meshgrid(x1,z1);
v_in = interp2(x,z,v,Xq,Yq);

%线性插值得到速度模型模型在坐标点的值
xx_in = round(x_train/0.001)+1;
zz_in = round(z_train/0.001)+1;
v_train = zeros(N_train,1);
U0_imag_train = zeros(N_train,1);
U0_real_train = zeros(N_train,1);

%% ANALYTICAL
% Distance from source to each point in the model
r = @(zz,xx)(zz.^2+xx.^2).^0.5;
vv = 1.83;

for is = 1:N_train
    
    f = f_train(is);
    omega = 1*2*pi*f;
    K = (omega./vv);
    G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));
    G_2D = (G_2D_analytic(z_train(is) - sz, x_train(is) - sx_train(is)))*7.7;
    v_train(is,1) = v_in(zz_in(is),xx_in(is));
    
    U0_real_train(is,1) = real(G_2D);
    U0_imag_train(is,1) = imag(G_2D);

end

m_train = 1./v_train.^2;
m0_train = ([(1./(v0).^2)]);

%%保存训练数据，坐标和模型参数
save sigsbee_train_data_fre_N50000_f510.mat U0_real_train U0_imag_train x_train sx_train z_train f_train m_train m0_train

%% Numerical results

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);
sx = (10:10:90)*dx; ns = length(sx);

[zz,xx] = ndgrid(z,x);
sx = repmat(sx,nx*nz,1);

x1 = xx(:); x_star = (repmat(x1,ns,1)); 
z1 = zz(:); z_star = (repmat(z1,ns,1));
sx_star = sx(:);

npmlz = 60; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;
NN = (Nz-2)*(Nx-2);

v0 = ones(n)*1.8300;

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);
v0_e=extend2d(v0,npmlz,npmlx,Nz,Nx);

src_x = 11:10:91;
src_z = 2;

Ps1 = getP_H(n,npmlz,npmlx,src_z,src_x);
Ps1 = Ps1'*12000;

[o,d,n] = grid2odn(z,x);
n=[n,1];

nb = [npmlz  npmlx 0];
n  = n + 2*nb;

f = 5; omega = 2*pi*f;
A = Helm2D((omega)./v_e(:),o,d,n,nb);
U  = A\Ps1;

K = (omega./vv);
G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));

A0 = Helm2D((2*pi*f)./v0_e(:),o,d,n,nb);
U0  = A0\Ps1;

for is = 1:ns

    U_2D = reshape(full(U(:,is)),[nz+2*npmlz,nx+2*npmlx]);
    U_2d = U_2D(npmlz+1:end-npmlz,npmlx+1:end-npmlx);
    
    xs = (src_x(is)-1)*dx;
    zs = (src_z-1)*dz;

    G_2D = (G_2D_analytic(zz - zs, xx - xs))*7.7;  
    
    G_2D(src_z,src_x(is)) = (G_2D(src_z-1,src_x(is)) + G_2D(src_z+1,src_x(is)) + G_2D(src_z,src_x(is)-1) + G_2D(src_z,src_x(is)+1))/4;
    dU_2d = U_2d-G_2D;
    
    du_real_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1) = real(dU_2d(:));
    du_imag_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1) = imag(dU_2d(:));
    
end

du_real_star(abs(du_real_star)>2)=0;
%%保存测试数据
save sigsbee_5Hz_testdata_fre.mat x_star sx_star z_star du_real_star du_imag_star
save du_real_star_5hz.mat du_real_star
save du_imag_star_5hz.mat du_imag_star

is = 5;
amp = 0.5;
figure;
pcolor(x,z,reshape(du_real_star(is*nz*nx+1:(is+1)*nz*nx),[nz,nx]));
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)
