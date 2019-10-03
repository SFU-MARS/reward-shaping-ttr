load('ttrValue.mat', 'ttrValue');
subplot(2,2,1);
contour(ttrValue(:,:,15), 'showtext', 'on');
xlabel('\theta=-\pi/4', 'fontsize', 15);
subplot(2,2,2);
contour(ttrValue(:,:,20), 'showtext', 'on');
xlabel('\theta=0','fontsize', 15);
subplot(2,2,3);
contour(ttrValue(:,:,31), 'showtext', 'on');
xlabel('\theta=\pi/2','fontsize', 15);
subplot(2,2,4);
contour(ttrValue(:,:,35), 'showtext', 'on');
xlabel('\theta=3\pi/4','fontsize', 15);
sup = suptitle("TTR function for simple car at four diffrent angles");
set(sup,'FontSize',15);


