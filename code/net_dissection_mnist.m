function net_dissection
   
        % choose an image from test images that is predicted correctly with
        % your network
        % This example is a zero digit test image from mnist (class 1)
   
        
        
        load('zero_digit.mat');
        
        % show the zero digit test image with larger size
        show = imresize(data,2);
        figure ;
        imshow(show);
        saveas(gcf , 'zero_digit.png');
        
      
        
        % load the final result of training
        % the training result is stored under 
        % code\mnist_data\mnist_baseline directory
        load('net-epoch-example-mnist.mat');
        
        % Pass the data through the trained net work 
        % and see what we get for high level and low level semantic
        % features
        % Build network exactly the same in cnn_mnist.m 
        layers = net.layers;
       
        % 1 conv1 
        w = layers{1}.weights{1} ;
        conv1 = vl_nnconv(data, w, [] , 'stride', 1, 'pad', 0) ;
        % 2 pool1
        pool1 = vl_nnpool(conv1, 2 , 'stride', 2, 'pad', 0 ) ;
        % 3 conv2
        w = layers{3}.weights{1} ;
        conv2 = vl_nnconv(pool1, w, [] ,  'stride', 1, 'pad', 0) ;
        % 4 pool2
        pool2 = vl_nnpool(conv2, 2 , 'stride', 2, 'pad', 0 ) ;
        % 5 conv3        
        w = layers{5}.weights{1} ;
        conv3 = vl_nnconv(pool2, w, [] ,  'stride', 1, 'pad', 0) ;
        % 6 relu 1
        relu1 = vl_nnrelu(conv3);
        % 7 conv4
        w = layers{7}.weights{1} ;
        conv4 = vl_nnconv(relu1, w, [] , 'stride', 1, 'pad', 0) ;
        % 8 softmax
        softmaxloss = vl_nnsoftmaxloss(conv4,1)
        
       
        cmp = jet(256) ;
      
        
        % plot the conv1
        figure;
        for ii = 1:size(conv1,3)
            subplot(4,5,ii);
            tmp = conv1(:,:,ii) ;
           imshow(ind2rgb(uint8(tmp),cmp));
        end
        saveas(gcf , 'first_layer_output.png');
        
        
        % plot the pool2 
        figure;
        
        for ii = 1:size(pool2,3)
            subplot(7,8,ii);
            tmp = pool2(:,:,ii) ;
            imshow(ind2rgb(uint8(tmp),cmp));
            
        end
        saveas(gcf , 'last_layer_output.png');
        
        % Try your network on Cifar data base

end