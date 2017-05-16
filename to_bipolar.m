function [bipolar_data] = to_bipolar(data,label)
    temp1 = zeros(length(data),length(label));
    for i = 1:length(data)
        for j=1:length(label)
            if label(j,1) == data(i,1)
                break;
            end
        end
        temp1(i,j) = 1;
    end
    bipolar_data = temp1*2-1;
    clear temp1;