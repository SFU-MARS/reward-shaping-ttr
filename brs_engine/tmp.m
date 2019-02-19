load('ttrX.mat');
load('ttrY.mat');
load('ttrW.mat');
load('ttrVxPhi.mat');
load('ttrVyPhi.mat');

full_ttrValue = zeros(41,41,41,41);

for x_idx = 1 : 41
    for y_idx = 1:41
        for vx_idx = 1 : 41
             for phi_idx = 1:41
            
            full_ttrValue(x_idx, y_idx,vx_idx, phi_idx) = max([ttrX(x_idx), ttrY(y_idx),ttrVxPhi(vx_idx, phi_idx)]);
                  
             end
           
        end
    end
end

[X,Y] = ndgrid(-5:0.25:5,-5:0.25:5);
Z = full_ttrValue(:,:,41,41);
surf(X,Y,Z);
hold on; 
