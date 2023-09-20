clear all
close all

sample = 3;
fre_num = 6;

load('bluewhitered.mat')

dx = 0.025; dz = dx;
nz = 101; nx = 101; n = [nz,nx];


wavefield_true = zeros(sample*nz,fre_num*nx);
wavefield_pred = zeros(sample*nz,fre_num*nx);

i = 0;
for ifre = [5, 6, 7, 8, 9,10];
    
    i = i + 1;
    str = ['du_real_pred_atan_',num2str(ifre)];
    filename=['' str 'hz_fre_l128_from5_ff01.mat'];
    load(filename);
    
    str = ['du_real_star_',num2str(ifre)];
    filename=['' str 'hz.mat'];
    load(filename);
    du_real_star(abs(du_real_star )>2)=0;
    j = 0;
    for is = [3, 5, 7];
        
        j = j + 1;
        a = (i-1)*nx + 1; 
        b = i*nx;
        c = (j-1)*nz + 1;
        d = j*nz;
        du_real_star_is = du_real_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);
        du_real_star_is2d = reshape(du_real_star_is,n);
        wavefield_true(c:d,a:b) = du_real_star_is2d;
        
        du_real_pred_is = du_real_pred( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);
        du_real_pred_is2d = reshape(du_real_pred_is,n);
        wavefield_pred(c:d,a:b) = du_real_pred_is2d;

    end
    
end

amp = 1;
figure;
pcolor(wavefield_true);
shading interp
axis ij
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
colormap(bluewhitered)
caxis([-amp amp])
set(gca,'FontSize',14)
% set(gca,'XAxisLocation','top')
colorbar
set(gcf, 'position', [0 0 1100 500]);

figure;
pcolor(wavefield_pred);
shading interp
axis ij
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
colormap(bluewhitered)
caxis([-amp amp])
set(gca,'FontSize',14)
% set(gca,'XAxisLocation','top')
colorbar
set(gcf, 'position', [0 0 1100 500]);

figure;
pcolor(wavefield_true-wavefield_pred);
shading interp
axis ij
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
colormap(bluewhitered)
caxis([-amp amp])
set(gca,'FontSize',14)
% set(gca,'XAxisLocation','top')
colorbar
set(gcf, 'position', [0 0 1100 500]);
