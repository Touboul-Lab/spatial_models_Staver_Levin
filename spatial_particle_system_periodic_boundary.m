%% Matrix simulation of Poisson Forest-Grass Markov Chain
% system is spatially extended with periodic boundary conditions (circle)
clear all;

% RNG seed
%rng(1546793)

Times=[];
DO_PLOT=0;
No_Abs=1;

% Spatial Parameters
L = 5; % working on [0,L]
dist = @(a,x) circ_dist(a,x,L);
kernel = @(x, a, sigma) exp( -( dist(a,x).^2)/(2*sigma^2))/(sqrt(2*pi)*sigma*(2*normcdf(L/(2*sigma))-1));

sigma_F = 0.05; % seed dispersal radius forest trees
sigma_W = 0.05; % fire radius

% Model Parameters
alpha = 0.5;

t2=0.4;
f0=0.1;
f1=0.9;
s2=0.05;

sites=3000;

P=1;                % Number of Patches
N=sites*ones(1,P);    % Number of Sites / Patch
NTot=sum(N);        % Total number of sites

T=250; % length of the simulation in real-time units
dt=0.01;
t0= 0;

MC=1;

J=alpha;
W=1;
phi=@(x) (f0+(f1-f0)./(1+exp(-(x-t2)/s2))); % sigmoid definition
Ntimes=length(t0:dt:(T-dt));

tic;

nbins=100;
Histo=zeros(nbins,MC);
p_grass=1;
p_no_grass=0;
Sol_Save=zeros(MC,sites,length(0:dt:T));
%% This block of code runs one simulation of the particle system using the Gillespie algorithm
for iteration=1:MC
    %fprintf('Monte Carlo Simulation #=%d/%d\n',iteration,MC);
    Solution=zeros(NTot,1);
    Locations=sort(L*rand(sites,1));
    [A,B]=meshgrid(Locations);
    J_Mat = L*kernel(A,B,sigma_F)/sites; % divide by number of sites for mean-field scaling
    W_Mat = L*kernel(A,B,sigma_W)/sites; % multiplication by L for normalization
    
%     % check that the normalization of the kernels is correct
%     locations_reg = L*(1/sites:1/sites:1);
%     [X,Y] = meshgrid(locations_reg);
%     %[X_L,Y_L] = meshgrid(locations_reg-L-1/sites);
%     %[X_R,Y_R] = meshgrid(L+ 1/sites + locations_reg);
%     J_Mat_regular = kernel(X,Y,sigma_F);% + J_F_fun(X,Y_L,sigma_F) + J_F_fun(X,Y_R,sigma_F) )/sites;
%     plot(sum(J_Mat_regular));
    
    Solution(:,1)=rand(NTot,1)<p_grass;
    No_grass=find((Locations>0.2*L).*(Locations<0.5*L));
    N_no_grass=length(No_grass);
    Solution(No_grass,1)=rand(N_no_grass,1)<p_no_grass;
    
    Times=[0];
    
    k=0;
    t=0;
    while (t<T)
        progressbar(t,T);
        k=k+1;
        BirthRates=alpha*((J_Mat*(1-Solution(:,k)))).*Solution(:,k);
        DeathRates=phi(W_Mat*Solution(:,k)).*(1-Solution(:,k));
        %BirthRates=alpha*((J_Mat*(1-Solution(:,k)))).*Solution(:,k);
        %DeathRates=phi(W_Mat*Solution(:,k)).*(1-Solution(:,k));
        
        totalIntensity=sum(BirthRates+DeathRates);
        
        NextEvent=-log(1-rand())/totalIntensity;
        
        t=t+NextEvent;
        if (t<T)
            Times(end+1)=t;
            
            CDF=cumsum(BirthRates+DeathRates)/totalIntensity;
            U=rand();
            i=1;
            while U>CDF(i)
                i=i+1;
            end
            Solution(:,k+1)=Solution(:,k);
            Solution(i,k+1)=1-Solution(i,k);
        else
            Times(end+1)=T;
            Solution(:,k+1)=Solution(:,k);
        end
    end
    %     Solution(:,end)=Solution(:,k);
    for i=1:sites
        Sol_Save(iteration,i,:)= interp1(Times,Solution(i,:),0:dt:Times(end));
    end
end
toc

%% Plots
figure(1);
imagesc(squeeze(mean(Sol_Save,1)));
custom_map = [1 1 1
    0 0.5 0];
colormap(custom_map);

Z=100;

V=squeeze(mean(Sol_Save,1));
U=squeeze(mean(reshape(V,sites/Z,Z,[]),1));
figure(2);
[x,y]=meshgrid(0:dt:Times(end),5*(1:Z)/Z);
pcolor(x,y,flipud(U))
shading interp;
custom_map = [
    linspace(1,0,100)' linspace(1,0.5,100)' linspace(1,0,100)'];
colormap(custom_map);