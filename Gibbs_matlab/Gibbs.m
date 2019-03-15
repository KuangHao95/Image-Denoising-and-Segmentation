function Gibbs(q,burnin,iteration)
% q = 0.7
% burnin = 4
% iteration = 10
    for j = 1:4
        % Read noise image: j_noise.txt without visualization
        [data,image] = read_data(strcat(num2str(j),'_noise.txt'),true,false,false);
        [height,width] = size(image);
        % Unitize the picture for building Ising model
        image(image == 0) = -1;
        image(image == 255) = 1;
        color = Denoise(image,q,burnin,iteration);
        % Convert to grey scale pic
        color(color >= 0) = 255;
        color(color < 0) = 0;
        count = 1;
        % Rewrite data
        for g = 1:width
            for h = 1:height
                data(count,3) = color(h,g);
                count = count + 1;
            end
        end
        write_data(data,strcat(num2str(j),'_denoise.txt'));
        read_data(strcat(num2str(j),'_denoise.txt'),true,false,true,strcat(num2str(j),'_denoise.jpg'));
    end
end

function ratio = Denoise(image,q,burn,iteration)
    [m,n] = size(image);
    external_factor = 0.5 * log(q / (1 - q));
    model = IsingModel(image,external_factor * image,3);
    
    avg = zeros(m,n);
    for i = 0 : (burn + iteration - 1)
        for x = 1 : m
            for y = 1 : n
                if rand(1) <= q
                    model.GibbsSample(x,y);
                end
            end
        end
        if i > burn
            avg = avg + model.img;
        end
    end
    ratio = avg / iteration;
end

