function [ttrXT, ttrYT, ttrV, ttrW] = ...
  Plane5D_approx_TTR(gMin, gMax, gN, valueXT, valueYT, valueV, valueW, tMax, interval)


    XTdims = [1 3];
    YTdims = [2 3];
    Vdim = 4;
    Wdim = 5;

    % Create grid structures for computation
    gXT = createGrid(gMin(XTdims), gMax(XTdims), gN(XTdims), 2);
    gYT = createGrid(gMin(YTdims), gMax(YTdims), gN(YTdims), 2);
    gV = createGrid(gMin(Vdim), gMax(Vdim), gN(Vdim));
    gW = createGrid(gMin(Wdim), gMax(Wdim), gN(Wdim));

    % Time horizon and intermediate results
    tau = 0:interval:tMax;


    ttrXT = TD2TTR(gXT, valueXT, tau);
    ttrYT = TD2TTR(gYT, valueYT, tau);
    ttrV  = TD2TTR(gV, valueV, tau);
    ttrW  = TD2TTR(gW, valueW, tau);


end
