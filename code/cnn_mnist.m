function [net, info] = cnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST


% record the time
tic
% data directory
opts.dataDir = fullfile('mnist_data','mnist') ;
% experiment result directory
opts.expDir = fullfile('mnist_data','mnist-baseline') ;
% image database
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
% set up the batch size (split the data into batches)
opts.train.batchSize = 100 ;
% number of Epoch (iterations)
opts.train.numEpochs = 10 ;
% resume the train
opts.train.continue = true ;
% use the GPU to train
opts.train.useGpu = false ;
% set the learning rate
opts.train.learningRate = 0.001 ;
% set weight decay
opts.train.weightDecay = 0.0005 ;
% set momentum
opts.train.momentum = 0.9 ;
% experiment result directory
opts.train.expDir = opts.expDir ;
% parse the varargin to opts. 
% If varargin is empty, opts argument will be set as above
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define a network similar to LeNet
f=1/100 ;

%% Define network 
net.layers = {} ;

% 1 conv1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;
% 2 pool1 (max pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0,...
                           'opts',{{}}) ;
% 3 conv2                      
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single')}}, ...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;
% 4 pool2 (max pool2)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0,...
                           'opts',{{}}) ;
% 5 conv3                      
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single')}}, ...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;
% 6 relu1
net.layers{end+1} = struct('type', 'relu','leak',0) ;
% 7 conv4
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,500,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'dilate', 1, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'opts',{{}}) ;
%8 softmax
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
vl_simplenn_display(net);
%% train the data
[net, info] = cnn_train_mnist(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
%% record the result into csv and plot the confusion matrix
load(['mnist_data/mnist-baseline/net-epoch-' int2str(opts.train.numEpochs) '.mat']);
load(['mnist_data/mnist-baseline/imdb' '.mat']);

fid = fopen('mnist_prediction.csv', 'w');
strings = {'ID','Label'};
for row = 1:size(strings,1)
    fprintf(fid, repmat('%s,',1,size(strings,2)-1), strings{row,1:end-1});
    fprintf(fid, '%s\n', strings{row,end});
end
fclose(fid);
ID = 1:10000;
dlmwrite('mnist_prediction.csv',[ID', info.val.prediction_class], '-append');
groundtruth = images.labels(60001:end);
prediction = info.val.prediction_class;
%generate confusion matrix
confusionMatrix = confusion_matrix(groundtruth , prediction);

cmp = jet(256);
figure ;
imshow(ind2rgb(uint8(confusionMatrix),cmp));
imwrite(ind2rgb(uint8(confusionMatrix),cmp) , 'mnist_confusion_matrix.png');
toc

% --------------------------------------------------------------------
% call back function get the part of the batch
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
% get image data base
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

mkdir(opts.dataDir) ;
% if the data directory does not exist (first time for the program)
for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end


f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

imdb.images.data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

 % produce groundTruth csv file
  
    fid = fopen('mnist_groundTruth.csv', 'w');
    strings = {'ID','Label'};
    for row = 1:size(strings,1)
        fprintf(fid, repmat('%s,',1,size(strings,2)-1), strings{row,1:end-1});
        fprintf(fid, '%s\n', strings{row,end});
    end
    fclose(fid);
    ID = 1:10000;
    ID = ID' ;
    output = zeros(10000,2);
    output(:,1) = ID ;
    output(:,2) = imdb.images.labels(60001:end)';
    dlmwrite ('mnist_groundTruth.csv',output, '-append');

