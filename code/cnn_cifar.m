% function are called with two types
% either cnn_cifar('coarse') or cnn_cifar('fine')
% coarse will classify the image into 20 catagories
% fine will classify the image into 100 catagories
function cnn_cifar(type, varargin)

if ~(strcmp(type, 'fine') || strcmp(type, 'coarse')) 
    error('The argument has to be either fine or coarse');
end

% record the time
tic
%% --------------------------------------------------------------------
%                                                         Set parameters
% --------------------------------------------------------------------
%
% data directory
opts.dataDir = fullfile('cifar_data','cifar') ;
% experiment result directory
opts.expDir = fullfile('cifar_data','cifar-baseline') ;
% image database
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
% set up the batch size (split the data into batches)
opts.train.batchSize = 100 ;
% number of Epoch (iterations)
opts.train.numEpochs = 30 ;
% resume the train
opts.train.continue = true ;
% use the GPU to train
opts.train.useGpu = false ;
% set the learning rate
opts.train.learningRate = [0.001*ones(1, 15) 0.0001*ones(1,8)  0.00001*ones(1, 7)]
% set weight decay
opts.train.weightDecay = 0.0005 ;
% set momentum
opts.train.momentum = 0.9 ;
% experiment result directory
opts.train.expDir = opts.expDir ;
% parse the varargin to opts. 
% If varargin is empty, opts argument will be set as above
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

imdb = load(opts.imdbPath) ;


%% Define network 
% The part you have to modify
net.layers = {} ;

% 1 conv1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{1e-4*randn(3,3,3,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'learningRate',[1,2],...
			   'dilate', 1, ...
                           'stride', 2, ...
                           'pad', 0,...
                           'opts',{{}}) ;

net.layers{end+1} = struct('type', 'relu','leak',0) ;

%net.layers{end+1} = struct('type', 'conv', ...
%                           'weights', {{1e-4*randn(3,3,32,32, 'single'), zeros(1, 32, 'single')}}, ...
%                           'learningRate',[1,2],...
%			   'dilate', 1, ...
%                           'stride', 1, ...
%                           'pad', 0,...
%                           'opts',{{}}) ;


% 2 pool1 (max pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;

%net.layers{end+1} = struct('type', 'dropout', 'rate', 0.25);

% 3 conv2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(3,3,32,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'learningRate',[1,2],...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;

% 4 relu2
%net.layers{end+1} = struct('type', 'relu','leak',0) ;

%net.layers{end+1} = struct('type', 'conv', ...
%                           'weights', {{0.01*randn(2,2,64,64, 'single'), zeros(1, 64, 'single')}}, ...
%                           'learningRate',[1,2],...
%                           'dilate', 1, ...
%                           'stride', 1, ...
%                           'pad', 0,...
%                           'opts',{{}}) ;

net.layers{end+1} = struct('type', 'relu','leak',0) ;

% 5 pool2 (avg pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0,...
                           'opts',{{}}) ; 

%net.layers{end+1} = struct('type', 'dropout', 'rate', 0.25);

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(3,3,32,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'learningRate',[1,2],...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;

                     
net.layers{end+1} = struct('type', 'relu','leak',0) ;


%net.layers{end+1} = struct('type', 'pool', ...
%                           'method', 'max', ...
%                           'pool', [3 3], ...
%                           'stride', 1, ...
%                           'pad', 0,...
%                           'opts',{{}}) ; 


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(3,3,32,64, 'single'), zeros(1, 64, 'single')}}, ...
                           'learningRate',[1,2],...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;

                     
net.layers{end+1} = struct('type', 'relu','leak',0) ;


%net.layers{end+1} = struct('type', 'pool', ...
%                           'method', 'max', ...
%                           'pool', [3 3], ...
%                           'stride', 1, ...
%                           'pad', 0,...
%                           'opts',{{}}) ; 

net.layers{end+1} = struct('type', 'dropout', 'rate', 0.2);

% 6 conv5
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.1*randn(2,2,64,100, 'single'), zeros(1, 100, 'single')}}, ...
                           'learningRate',[1,2],...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;

%net.layers{end+1} = struct('type', 'relu','leak',0) ;

% 7 loss
net.layers{end+1} = struct('type', 'softmaxloss') ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end
%% display the net
vl_simplenn_display(net, 'inputSize', [32 32 3 25]);
%% start training
[net,info] = cnn_train_cifar(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 2) , 'test', find(imdb.images.set == 3)) ;
%% Record the result into csv and draw confusion matrix
load(['cifar_data/cifar-baseline/net-epoch-' int2str(opts.train.numEpochs) '.mat']);
load(['cifar_data/cifar-baseline/imdb' '.mat']);
fid = fopen('cifar_prediction.csv', 'w');
strings = {'ID','Label'};
for row = 1:size(strings,1)
    fprintf(fid, repmat('%s,',1,size(strings,2)-1), strings{row,1:end-1});
    fprintf(fid, '%s\n', strings{row,end});
end
fclose(fid);
ID = 1:numel(info.test.prediction_class);
dlmwrite('cifar_prediction.csv',[ID', info.test.prediction_class], '-append');

val_groundtruth = images.labels(45001:end);
val_prediction = info.val.prediction_class;
val_confusionMatrix = confusion_matrix(val_groundtruth , val_prediction);
cmp = jet(50);
figure ;
imshow(ind2rgb(uint8(val_confusionMatrix),cmp));
imwrite(ind2rgb(uint8(val_confusionMatrix),cmp) , 'cifar_confusion_matrix.png');
toc

% --------------------------------------------------------------------
%% call back function get the part of the batch
function [im, labels] = getBatch(imdb, batch , set)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
% data augmentation
%if set == 1 % training
    % fliplr
        
    % noise
        
    % random crop
    
    % and other data augmentation
%end


%if set ~= 3
%    labels = imdb.images.labels(1,batch) ;
%end

% data augmentation
if set == 1 % training
    total = size(batch, 2);
    % fliplr
    im_1 = zeros(size(im));
    for i = 1:total
        p = rand(1);
        if (p > 0.2)
            im(:,:,:,i) = fliplr(im(:,:,:,i));
        end
    end
     % rotate
     for i = 1:total
         p = rand(1);
         if (p > 0.2)
             im_2(:,:,:,i) = imrotate(im(:,:,:,i), 10, 'crop');
         end
     end            
     % random crop
     for i = 1:total
         p = rand(1);
         if (p > 0.2)
             x = randi(round(0.1 * size(im, 1)));
             y = randi(round(0.1 * size(im, 2)));
             tmp = imresize(im(:,:,:,i), 1.1);
             tmp = imcrop(tmp, [x y size(im,1)-1 size(im,2)-1]);
             im_3(:,:,:,i) = tmp;
         end
     end      
     
     % and other data augmentation
end


if set ~= 3
    labels = imdb.images.labels(1,batch) ;
end

