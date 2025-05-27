function duqu(imgDataPath,save_path)
    imgDataDir = dir(imgDataPath); % 遍历所有文件
    for i = 3:length(imgDataDir)
        imgDir = dir(strcat(strcat(imgDataPath,"/"),imgDataDir(i).name));
        for j = 3:length(imgDir) % 遍历所有图片
            image_path = strcat('',strcat(strcat(strcat(strcat(imgDataPath,"/"),imgDataDir(i).name),strcat("/",imgDir(j).name)),''));
            disp(image_path);
                f=imread(image_path);
                hist_image = rgb_eq(f);
                save_hist_path = strcat('',strcat(strcat("E:/hist","/"),strcat(strcat(num2str(j),imgDir(j).name)),''));
                imwrite(hist_image,save_hist_path);
                btws_image = btws(save_hist_path,57);
                %btws_image = imsubtract(f,uint8(btws_image));
                save_btws_path = strcat('',strcat(strcat(strcat(strcat(save_path,"/"),imgDataDir(i).name),strcat("/",imgDir(j).name)),''));
                %save_btws_path = strcat('',strcat(strcat(strcat(save_path,"/"),imgDir(j).name),''));
                imwrite(btws_image,save_btws_path);
                disp(save_btws_path);
            
        end
    end
end

