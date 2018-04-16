x = imread('./telhados.png');
x=  x(:,:,2);
sobel_v = [-1 -2 -1; 0 0 0; 1 2 1];
sobel_h = [-1 0 1; -2 0 2; -1 0 1];

borda_v = conv2(x,sobel_v);
borda_h = conv2(x,sobel_h);
borda_v = borda_v(2:end-1,2:end-1);
borda_h = borda_h(2:end-1,2:end-1);

figure
subplot(1,2,1); imshow(uint8(x)); title('Original')
% subplot(1,2,1); imshow(uint8(borda_v)); title('Bordas verticais da imagem')
% subplot(1,2,2); imshow(uint8(borda_h)); title('Bordas horizontais da imagem')

% figure()
% imshow(d)