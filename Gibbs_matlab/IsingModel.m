classdef IsingModel < handle
    properties
        img;
        ext_factor;
        beta;
    end
    
    methods
        function obj = IsingModel(i,e,b)
           obj.img = i;
           obj.ext_factor = e;
           obj.beta = b;
        end
        % Get 4 neighbors of point(x,y)
        function n_value = neighbor(obj,x,y)
            [height,width] = size(obj.img);
            %n = [];
            n_value = [];
            if x == 1
                %n = [n,[height,y]];
                n_value = [n_value,obj.img(height,y)];
            else
                %n = [n,[x-1,y]];
                n_value = [n_value,obj.img(x-1,y)];
            end
            if x == height
                %n = [n,[1,y]];
                n_value = [n_value,obj.img(1,y)];
            else
                %n = [n,[x + 1,y]];
                n_value = [n_value,obj.img(x + 1,y)];
            end
            
            if y == 1
                %n = [n,[x,width]];
                n_value = [n_value,obj.img(x,width)];
            else
                %n = [n,[x,y - 1]];
                n_value = [n_value,obj.img(x,y - 1)];
            end
            if y == width
                %n = [n,[x,1]];
                n_value = [n_value,obj.img(x,1)];
            else
                %n = [n,[x,y + 1]];
                n_value = [n_value,obj.img(x,y + 1)];
            end
        end
        % Compute local energy of point(x,y)
        function energy = LocalEnergy(obj,x,y)
            energy = obj.ext_factor(x,y) + sum(obj.neighbor(x,y)) ;
        end
        % Use Gibbs sampling
        function GibbsSample(obj,x,y)
           p = 1 / (1 + exp(-2 * obj.beta * LocalEnergy(obj,x,y)));
           if rand(1) <= p
               obj.img(x,y) = 1;
           else
               obj.img(x,y) = -1;
           end
        end
    end
end