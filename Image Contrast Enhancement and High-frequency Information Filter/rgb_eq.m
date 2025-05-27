function result = rgb_eq(img)
    if length(size(img))>1
        rimg = img(:,:,1);  
        gimg = img(:,:,1);  
        bimg = img(:,:,1);  
        resultr = adapthisteq(rimg,'NumTiles',[5 5]);  
        resultg = adapthisteq(gimg,'NumTiles',[5 5]);  
        resultb = adapthisteq(bimg,'NumTiles',[5 5]);  
        
    result = cat(3, resultr, resultg, resultb); 
    end
end