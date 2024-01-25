function result = S(I,window, loc)
    if loc == 'lt'
        % 窓内の左上をとる  
        result = I(1:window:size(I,1), 1:window:size(I,2),:);
    elseif loc == 'c'
        % 窓内の真ん中をとる  
        result = I(floor(window/2):window:size(I,1), floor(window/2):window:size(I,2),:);
    end
end