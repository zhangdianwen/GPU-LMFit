%  To test GPU_LMFit:
%
% 		A GPU-based parallel Levenberg–Marquardt least-squares minimization fitting library 
% 		in CUDA C/C++ 
% 
% 		This software was developed based on the C version MPFit (see References).
% 		Parallel computation algorithms and translation to CUDA C/C++ were applied by Xiang 
% 		Zhu and Dianwen Zhang, at Image Technology Group, Beckman Institute for Advanced 
% 		Science & Technology, University of Illinois at Urbana-Champaign. Our contact 
% 		information can be found at http://itg.beckman.illinois.edu/.
%
%  Function of this program:
%       To study the fit accuracy of two-dimenstional Gaussian fit using GPU_LMFit 
%

clear all
close all
clc

% Which version of Smith MLE programs to use?
% 1 - to use GPUgaussMLE.mexw32 (version 1.0) in 32 bits system;  
% 2 - to use GPUgaussMLEv2.mexw32 (version 2.0) in 32 bits system; (Note: CRLB might be wrong for large number of photons)
% 3 - to use GPUgaussMLEv2.mexw64 (version 2.0) in 64 bits system; (Note: CRLB might be wrong for large number of photons)
SmithMLEVer = 1; 
                    
% GPU CUDA setting
GPU_Block_Size = 128; 
GPU_Grid_Size = 1024; 

% Other settings
NumOfFits = 5000; %number of images to fit
bl = 0;           %Base line
bg = 2;           %background fluorescence in photons/pixel/frame
ImgDim = 7;       %linear size of fit region in pixels.
sigmax = 1.0;
sigmay = sigmax;
n = 5;

Init_sigmax = sigmax;
Init_sigmay = sigmay;
init_ins = Init_sigmax;

Smith_init_ins = [Init_sigmax Init_sigmay];

NPhotonsMax = 10000;
NPhotonsMin = 100;
Npoints = 10;
LogNPhotonsMax = log10(NPhotonsMax);
LogNPhotonsMin = log10(NPhotonsMin);
LogNPhotons = linspace(LogNPhotonsMin,LogNPhotonsMax,Npoints);

x00 = ImgDim/2;
y00 = x00;
dx0 = 0;
for jj = 1:Npoints
    
    NPhotons(jj) = 10^LogNPhotons(jj);   %expected photons/frame
    fprintf('\n\n\nNPhotons = %g :\n',NPhotons(jj));
    A = NPhotons(jj)/2/pi/sigmax/sigmay;
    
    %generate a stack of images
    coords=[x00+dx0*ones(NumOfFits,1) y00+zeros(NumOfFits,1)];
    
    % PSF - point spread function
    [xi,yj] = meshgrid(1:ImgDim,1:ImgDim);
    xi = repmat(xi-1, [1 1 NumOfFits]);
    yj = repmat(yj-1, [1 1 NumOfFits]);
    x0=repmat(shiftdim(coords(:,1),-2),[ImgDim,ImgDim,1]);
    y0=repmat(shiftdim(coords(:,2),-2),[ImgDim,ImgDim,1]);
    ImgData = NPhotons(jj)/(2*pi*sigmax*sigmay)*exp(-1/2*((xi-x0)/sigmax).^2)...
        .*exp(-1/2*((yj-y0)/sigmay).^2) + bg;
    
    if(jj==1)
        figure('Name', 'Sample image')
        subplot(1,2,1)
        imagesc(ImgData(:,:,1))
        axis image
        title('Model')
        drawnow
    end
    
    % Add Poisson noise to images using poissrnd
    ImgData = poissrnd(ImgData,ImgDim,ImgDim,NumOfFits);
    ImgData = ImgData + bl;
    
    if(jj==1)
        subplot(1,2,2)
        imagesc(ImgData(:,:,1))
        axis image
        title('Noised')
        drawnow
    end
       
    
    %% Single precision GPU_LMFit MLE
    tic
    [FittedX InfoNum_RS] = GPU2DGaussFit_RS(single(reshape(permute(ImgData,[2,1,3]),1,ImgDim*ImgDim*NumOfFits)), ...
        ImgDim, init_ins, 0, 0, 0, GPU_Block_Size, GPU_Grid_Size);
    GPU_LMFit_MLE_RS_t(jj)=toc;
    
    GPU_LMFit_MLE_RS_NofFits(jj) = NumOfFits/GPU_LMFit_MLE_RS_t(jj);
    fprintf('\nSingle precision GPU_LMFit GPU2DGaussFit has performed %g fits per second.\n', GPU_LMFit_MLE_RS_NofFits(jj));
    GPU_LMFit_MLE_RS.B = FittedX(1:n:end);
    GPU_LMFit_MLE_RS.A = FittedX(2:n:end);
    GPU_LMFit_MLE_RS.x0 = FittedX(3:n:end);
    GPU_LMFit_MLE_RS.y0 = FittedX(4:n:end);
    GPU_LMFit_MLE_RS.sx = FittedX(5:n:end);
    GPU_LMFit_MLE_RS_stdx0(jj)=std(GPU_LMFit_MLE_RS.x0'-coords(:,1));
    GPU_LMFit_MLE_RS_stdy0(jj)=std(GPU_LMFit_MLE_RS.y0'-coords(:,2));
    fprintf('Single precision GPU2DGaussFit gives the x0 precision of %g.\n', GPU_LMFit_MLE_RS_stdx0(jj));
    fprintf('Single precision GPU2DGaussFit gives the y0 precision of %g.\n', GPU_LMFit_MLE_RS_stdy0(jj));
     
     %% Double precision GPU_LMFit MLE
    tic
    [FittedX InfoNum_RD] = GPU2DGaussFit_RD(reshape(permute(ImgData,[2,1,3]),1,ImgDim*ImgDim*NumOfFits), ...
        ImgDim, init_ins, 0, 0, 0, GPU_Block_Size, GPU_Grid_Size);
    GPU_LMFit_MLE_RD_t(jj)=toc;
    
    GPU_LMFit_MLE_RD_NofFits(jj) = NumOfFits/GPU_LMFit_MLE_RD_t(jj);
    fprintf('\nDouble precision GPU_LMFit GPU2DGaussFit has performed %g fits per second.\n', GPU_LMFit_MLE_RD_NofFits(jj));
    GPU_LMFit_MLE_RD.B = FittedX(1:n:end);
    GPU_LMFit_MLE_RD.A = FittedX(2:n:end);
    GPU_LMFit_MLE_RD.x0 = FittedX(3:n:end);
    GPU_LMFit_MLE_RD.y0 = FittedX(4:n:end);
    GPU_LMFit_MLE_RD.sx = FittedX(5:n:end);
    GPU_LMFit_MLE_RD_stdx0(jj)=std(GPU_LMFit_MLE_RD.x0'-coords(:,1));
    GPU_LMFit_MLE_RD_stdy0(jj)=std(GPU_LMFit_MLE_RD.y0'-coords(:,2));
    fprintf('Double precision GPU_LMFit GPU2DGaussFit gives the x0 precision of %g.\n', GPU_LMFit_MLE_RD_stdx0(jj));
    fprintf('Double precision GPU_LMFit GPU2DGaussFit gives the y0 precision of %g.\n', GPU_LMFit_MLE_RD_stdy0(jj));
   
    %% Smith's MLE fit
    % Does Smith MLE exist?
    switch SmithMLEVer
        case 1
            if exist('GPUgaussMLE.mexw32', 'file') && strcmp(mexext, 'mexw32')
                NoSmithMLE = 0;
            else
                NoSmithMLE = 1;
                fprintf('\n\n... Can NOT find Smith\''s MLE GPUgaussMLE.mexw32 to compare! ...\n\n')
            end
        case 2
            if exist('GPUgaussMLEv2.mexw32', 'file') && strcmp(mexext, 'mexw32')
                NoSmithMLE = 0;
            else
                NoSmithMLE = 1;
                fprintf('\n\n... Can NOT find Smith\''s MLE GPUgaussMLEv2.mexw32 to compare! ...\n\n')
            end
        case 3
            if exist('GPUgaussMLEv2.mexw64', 'file') && strcmp(mexext, 'mexw64')
                NoSmithMLE = 0;
            else
                NoSmithMLE = 1;
                fprintf('\n\n... Can NOT find Smith\''s MLE GPUgaussMLEv2.mexw64 to compare! ...\n\n')
            end
        otherwise
            NoSmithMLE = 1;
    end
    if(NoSmithMLE==0)
        fprintf('\n\n');
        
        % Parameters for Smith MLE programs
        SmithGPUMLE_Iters = 20;
        SmithGPUMLE_FitType = 1;    %   1 - position,bg,N only;
        %   2 - also PSF sigma;
        %   3 - z position;
        %   4 - PSF sigma_x,sigma_y (CRLB not returned).
        %   Default=1
        
        switch SmithMLEVer
            case 1
                tic;
                [SmithMEL_x0 SmithMEL_y0 SmithMEL_N SmithMEL_BG SmithMEL_S ...
                    CRLBx CRLBy CRLBn CRLBb CRLBs LogL] = ...
                    GPUgaussMLE(permute(single(ImgData),[2 1 3]),Smith_init_ins, SmithGPUMLE_Iters, SmithGPUMLE_FitType);
                SmithMLE_t(jj)=toc;
                SmithMLE_NofFits(jj) = NumOfFits/SmithMLE_t(jj);
                fprintf('\nGPUgaussMLE ver 1.0 has performed %g fits per second.\n', SmithMLE_NofFits(jj));
                
                %convert variances to standard deviations
                CRLBx=sqrt(CRLBx);
                CRLBy=sqrt(CRLBy);
                CRLBn=sqrt(CRLBn);
                CRLBb=sqrt(CRLBb);
                CRLBs=sqrt(CRLBs);
                
                %report some details
                SmithMLE_stdx0(jj)=std(SmithMEL_x0-coords(:,1));
                SmithMLE_stdy0(jj)=std(SmithMEL_y0-coords(:,2));
                SmithMLEs(jj) = mean(SmithMEL_S);
                SmithMLEN(jj) = mean(SmithMEL_N);
                meanCRLBx(jj)= mean(CRLBx);
                meanCRLBy(jj)= mean(CRLBy);
            case {2,3}
                tic
                [P CRLB LL]=GPUgaussMLEv2(permute(single(ImgData),[2 1 3]),Smith_init_ins, ...
                    SmithGPUMLE_Iters, SmithGPUMLE_FitType);
                SmithMLE_t(jj)=toc;
                SmithMLE_NofFits(jj) = NumOfFits/SmithMLE_t(jj);
                fprintf('\nGPUgaussMLE ver 2.0 has performed %g fits per second.\n', SmithMLE_NofFits(jj));
                
                %convert variances to standard deviations
                CRLB=sqrt(CRLB);
                
                %report some details
                SmithMLE_stdx0(jj)=std(P(:,1)-coords(:,1));
                SmithMLE_stdy0(jj)=std(P(:,2)-coords(:,2));
                meanCRLBx(jj)=mean(CRLB(:,1));
                meanCRLBy(jj)=mean(CRLB(:,2));
        end
        
        fprintf('The standard deviation of x-position error is %g \n',SmithMLE_stdx0(jj))
        fprintf('The standard deviation of y-position error is %g \n',SmithMLE_stdy0(jj))
        fprintf('The mean returned CRLB based x-position precision is %g \n',meanCRLBx(jj))
        fprintf('The mean returned CRLB based y-position precision is %g \n',meanCRLBy(jj))
    else
        SmithMLE_t(jj) = 0;
        SmithMLE_stdx0(jj) = NaN;
        SmithMLE_stdy0(jj) = NaN;
        meanCRLBx(jj)= NaN;
        meanCRLBy(jj)= NaN;
    end
end


% Save data to File
% TimeStr = datestr(now, 'yyyy_mm_dd_HH_MM_SS_AM');
% GPU_LMFit_MLE_RS_Speed = (NumOfFits./GPU_LMFit_MLE_RS_t);
% GPU_LMFit_MLE_RD_Speed = (NumOfFits./GPU_LMFit_MLE_RD_t);
% Smith_MLE_Speed = (NumOfFits./SmithMLE_t);
% Result2File = [NPhotons' SmithMLE_stdx0' SmithMLE_stdy0' GPU_LMFit_MLE_RS_stdx0' ...
%     GPU_LMFit_MLE_RS_stdy0' GPU_LMFit_MLE_RD_stdx0' GPU_LMFit_MLE_RD_stdy0' ...
%     meanCRLBx' meanCRLBy'];
% Result2File = double(Result2File);
% outputfilename = [mfilename('fullpath') '_stdx0y0@' TimeStr '.txt'];
% save(outputfilename, 'Result2File', '-ASCII');


%%Display Figures
figure('Name', 'Accuracy of SmithMLE and GPU2DGaussFit')
subplot(1,2,1)
loglog(NPhotons, SmithMLE_stdx0, 'ro', ...
    NPhotons, GPU_LMFit_MLE_RS_stdx0, 'bx',...
    NPhotons, GPU_LMFit_MLE_RD_stdx0, 'g+',...
    NPhotons, meanCRLBx, 'r-');
axis tight
title('Experimental-x0');
subplot(1,2,2)
loglog(NPhotons, SmithMLE_stdy0, 'ro', ...
    NPhotons, GPU_LMFit_MLE_RS_stdy0, 'bx',...
    NPhotons, GPU_LMFit_MLE_RD_stdy0, 'g+',...
    NPhotons, meanCRLBy, 'r-');
axis tight
title('Experimental-y0');
legend('SmithMLE','GPU\_LMFit\_RS\_MLE','GPU\_LMFit\_RD\_MLE','CRLB');
