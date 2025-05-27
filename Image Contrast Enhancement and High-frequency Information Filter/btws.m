function [btws_image] = btws(image,Q)
image=imread(image);
f1 = image(:,:,1); %将rgb图像转换成灰度图像
f2 = image(:,:,2);
f3 = image(:,:,3);
[M, N] = size(f1);
a1 = fft2(f1);
a1 = fftshift(a1);
m1 = fix(M / 2); n1 = fix(N / 2);
for u = 1:M
    for v = 1:N
        D1 = sqrt((u - m1)^2 + (v - n1)^2);
        if D1 == 0
            H1(u, v) = 0;
        else
            %    H(u,v)=1/(1+0.414*(500/D1)^4);%截至频率为500
            H1(u, v) = 1 / (1 + (Q / D1)^4); %2阶巴特沃斯高通滤波器，截至频率为500
        end  
    end
end
F1 = H1 .* a1;
F1 = ifftshift(F1);
I1 = abs(ifft2(F1));

a2 = fft2(f2);
a2 = fftshift(a2);
m1 = fix(M / 2); n1 = fix(N / 2);

for u = 1:M
    for v = 1:N
        D2 = sqrt((u - m1)^2 + (v - n1)^2);
        if D2 == 0
            H2(u, v) = 0;
        else
            %    H(u,v)=1/(1+0.414*(500/D1)^4);%截至频率为500
            H2(u, v) = 1 / (1 + (Q / D2)^4); %2阶巴特沃斯高通滤波器，截至频率为500
        end  
    end
end
F2 = H2 .* a2;
F2 = ifftshift(F2);
I2 = abs(ifft2(F2));


a3 = fft2(f3);
a3 = fftshift(a3);
m1 = fix(M / 2); n1 = fix(N / 2);

for u = 1:M
    for v = 1:N
        D3 = sqrt((u - m1)^2 + (v - n1)^2);
        if D3 == 0
            H3(u, v) = 0;
        else
            %    H(u,v)=1/(1+0.414*(500/D1)^4);%截至频率为500
            H3(u, v) = 1 / (1 + (Q / D3)^4); %2阶巴特沃斯高通滤波器，截至频率为500
        end  
    end
end
F3 = H3 .* a3;
F3 = ifftshift(F3);
I3 = abs(ifft2(F3));

btws_image(:,:,1) = I1;
btws_image(:,:,2) = I2;
btws_image(:,:,3) = I3;



