import mxnet as mx
import pandas as pa
import numpy as np
# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)



def myNN(datapath,net,batch_size,epochs, learningrate):

    #create network Symbol(just a graph no any space)
    # net = createNetWorkL3(10,20,10)
    #  initial the element in the network
    #  the initial is not only give the first weight and bias, we need tell the computer to
    #  allocate the memory for the element in the network
    #  so what i can do :  SimpleBind let mxnet give related memory for network

    #download data to mxnet.io.NDIter
    train_iter, test_iter = loaddata2DataIter(datapath,batchsize=batch_size)
    train_info = train_iter.provide_data+train_iter.provide_label
    input_info = dict(train_info)  #data:shape;softmax_label:shape

    # allocate memory by Simplebind in net
    '''
    Bind current symbol to get an executor, allocate all the ndarrays needed.
    Allows specifying data types.
    This function will ask user to pass in ndarray of position they like to bind to,
    and it will automatically allocate the ndarray for arguments
    and auxiliary states that user did not specify explicitly.

    Return type:	mxnet.Executor
    '''
    exe = net.simple_bind(ctx=mx.cpu(), **input_info)

    #initial weight and bias
    #combile network arguments and simple_bind allocate array
    exe_info = dict(zip(net.list_arguments(), exe.arg_arrays))
    print 'provide len',train_iter.provide_data[0],train_iter.provide_label[0]
    # data = exe_info[train_iter.provide_data[0][0]]
    #
    # label = exe_info[train_iter.provide_label[0][0]]
    # print data,label
    init = mx.init.Uniform(scale=0.08)
    for i,j in exe_info.items():
        if i not in input_info:
            init(i,j)

    #update
    updater = update_SGD(learningrate,train_iter)
    #caculate mericy
    metric = mx.metric.Accuracy()
    # train loop
    for epoch in range(epochs):
        train_iter.reset()
        metric.reset()
        num = 0
        for batch in train_iter:
            # data[:]= batch.data[0]
            # label[:] = batch.label[0]
            exe_info['data'][:]=batch.data[0]     #[:] is very important but why
            # print 'batch.data shape', batch.data.__len__()  # batch.data shape 1
            # print 'batch.data[0]',batch.data[0]
            # print 'exe_info[\'data\']',exe_info['data']
            exe_info['softmax_label'][:]= batch.label[0]
            exe.forward(is_train=True)
            exe.backward()
            #update
            for index, args in enumerate(zip(exe.arg_arrays,exe.grad_arrays)):
                weight, grad = args
                updater(index,grad,weight)#update each weight and grad
            metric.update(batch.label,exe.outputs)
            num += 1
            # if num % 60 == 0:
                # print 'epoch',epoch,'iter',num,'metric',metric.get()

        for batch in test_iter:
            exe_info['data'][:] = batch.data[0]
            exe_info['softmax_label'][:] = batch.label[0]
            exe.forward()
            metric.update(batch.label,exe.outputs)
        print 'epoch',epoch,'test',metric.get()



def load_data(train_path):
    '''
    path :"/home/xuqian/mxnet2/data/train.csv"
    :param train_path: train_csv path
    :return: train_array([6000,(28,28)],[6000,1]) , test_array is the same as the tuple

    '''
    train_csv = pa.read_csv(train_path)

    # train data
    train_data = np.array(train_csv[0:6000])
    train_label = []    #save the label
    train_dim = []      # save image data which 1*784
    for i in train_data:
        # outlabel = changelabel(i[0]) # change type to 10-dim
        outlabel =i[0]
        train_label.append(outlabel)
        train_dim.append(i[1:])
    # # X = X.astype(np.float32) / 255
    # train_dim = train_dim.astype()

    # test data
    test_data = np.array(train_csv[-4000:])
    test_label = []   #save the test label
    test_dim = []      # save image data
    for j in test_data:
        # outlabel = changelabel(j[0])
        outlabel = j[0]
        test_label.append(outlabel)
        test_dim.append(j[1:])


    train_d = mx.nd.array(train_dim)
    train_d = train_d.astype(np.float32)/255
    train_l = mx.nd.array(train_label)
    test_d = mx.nd.array(test_dim)
    test_d = test_d.astype(np.float32) / 255
    test_l = mx.nd.array(test_label)

    return train_d,train_l,test_d,test_l


def changelabel(label):
    '''
    there is no used the changtype
    :param label: raw data label
    :return: output_layer type that one element is 1 and other is 0
    '''
    out_label = np.zeros(10)
    out_label[label]=1
    return out_label


def update_SGD(learnigrate,train_iter):
    opt_SGD = mx.optimizer.SGD(
        learning_rate=learnigrate,
        momentum=0.8,
        wd=0.00001,
        rescale_grad=1.0/(train_iter.batch_size )#True or False
    )
    update = mx.optimizer.get_updater(opt_SGD)
    return update
    # return opt_SGD



def loaddata2DataIter(datapath,batchsize):

    '''
    :param datapath: all data path
    :param batchsize:
    :return: train_iter, test_iter
    '''

    train_d, train_l, test_d, test_l = load_data(datapath)
    train_iter = mx.io.NDArrayIter(train_d, train_l, batch_size=batchsize)
    test_iter = mx.io.NDArrayIter(test_d, test_l, batch_size=batchsize)
    # test_iter.
    # mxnet.io.NDArrayIter encapsulate the data ,help us get information from data easily
    return train_iter, test_iter



def createNetWorkL1(hidden):
    '''
       create a one layer network,
    :param hidden: fullyconnected layer hidden_num
    :return:
    '''

    imput_layer = mx.sym.Variable('data')
    # output_layer = mx.sym.Variable('out')
    # layer1: fullyconnected
    fc = mx.sym.FullyConnected(data=imput_layer, name='fullconnectlayer', num_hidden=hidden)
    act = mx.sym.Activation(data=fc, name='actlayer', act_type='relu')
    net = mx.sym.SoftmaxOutput(data=act, name='softmax')
    return net


def createNetWorkL3(h1,h2,h3):
    '''
       create a one layer network,
    :param hidden: fullyconnected layer hidden_num
    :return:
    '''

    imput_layer = mx.sym.Variable('data')
    # output_layer = mx.sym.Variable('out')
    # layer1: fullyconnected
    fc1 = mx.sym.FullyConnected(data=imput_layer, name='fc1', num_hidden=h1)
    act1 = mx.sym.Activation(data=fc1, name='act1r', act_type='relu')
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=h2)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type='relu')
    fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=h3)
    act3 = mx.sym.Activation(data=fc3, name='act3', act_type='relu')
    net = mx.sym.SoftmaxOutput(data=act3, name='softmax')
    return net



def NN1(datapath,batch_size,hidden, epoch, learningrate):
    '''
    The layer just have one (FullConnected layer and related activation )and have a output layer
    :param datapath:
    :param batch_size:
    :param hidden:
    :param epoch:
    :param learningrate:
    :return:
    '''
    net = createNetWorkL1(hidden);
    #load data
    train_iter, test_iter = loaddata2DataIter(datapath)

    # train a model
    model = mx.model.FeedForward(
        ctx=mx.cpu(),  # Run on GPU 0
        symbol=net,  # Use the network we just defined
        num_epoch=epoch,  # Train for 10 epochs
        learning_rate=learningrate,  # Learning rate
        momentum=0.9,  # Momentum for SGD with momentum
        wd=0.00001)  # Weight decay for regularization
    model.fit(
        X=train_iter,  # Training data set
        eval_data=test_iter, # Testing data set. MXNet computes scores on test set every epoch
        batch_end_callback=mx.callback.Speedometer(batch_size, 200)
        # mx.callback.Speedometer()
    )  # Logging module to print out progress

    print 'final result is :',learningrate, model.score(test_iter)*100,'%'
    return model




def NN3(datapath,batch_size,hidden1,hidden2,hidden3, epoch, learningrate):
    '''
    The layer just have one (FullConnected layer and related activation )and have a output layer
    :param datapath:
    :param batch_size:
    :param hidden:
    :param epoch:
    :param learningrate:
    :return:
    '''
    ## create a one-layer MLP
    imput_layer = mx.sym.Variable('data')
    # output_layer = mx.sym.Variable('out')
    # layer1: fullyconnected
    fc1 = mx.sym.FullyConnected(data=imput_layer, name='fc1', num_hidden=hidden1)
    act1 = mx.sym.Activation(data=fc1, name='act1r', act_type='relu')
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=hidden2)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type='relu')
    fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=hidden3)
    act3 = mx.sym.Activation(data=fc3, name='act3', act_type='relu')
    net = mx.sym.SoftmaxOutput(data=act3, name='softmax')
    # mx.viz.plot_network(net)
    #load data
    train_data,train_label,test_data,test_label = load_data(datapath)
    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(test_data, test_label, batch_size=batch_size)


    # train a model
    model = mx.model.FeedForward(
        ctx=mx.cpu(),  # Run on GPU 0
        symbol=net,  # Use the network we just defined
        num_epoch=epoch,  # Train for 10 epochs
        learning_rate=learningrate,  # Learning rate
        momentum=0.9,  # Momentum for SGD with momentum
        wd=0.00001)  # Weight decay for regularization
    model.fit(
        X=train_iter,  # Training data set
        eval_data=test_iter, # Testing data set. MXNet computes scores on test set every epoch
        batch_end_callback=mx.callback.Speedometer(batch_size, 200)
        # mx.callback.Speedometer()
    )  # Logging module to print out progress

    print 'final result is :',learningrate, model.score(test_iter)*100,'%'
    # return model


if __name__ == "__main__":
    datapath = "/home/xuqian/mxnet2/data/train.csv"
    batch_size = 60
    hidden = 100
    h1 = 300
    h2 = 300
    h3 = 128
    epoch = 60
    # train_iter, test_iter = loaddata2DataIter(datapath, 60)
    # print train_iter.provide_data
    # print train_iter.provide_label
    # print test_iter.provide_label
    # learningrate = 0.00005
    # learningrate = 0.001
    # learningrate = 0.01
    learningrate = 0.1
    net = createNetWorkL3(h1=h1,
                          h2=h2,
                          h3=h3
                          )
    # epoch  8  test('accuracy', 0.9996666666666667)
    # epoch  9  test('accuracy', 1.0)
    myNN(
        net=net,
        datapath=datapath,
        batch_size=batch_size,
        epochs=epoch,
        learningrate=learningrate
    )

    # model = CustomNN(datapath,batch_size,hidden,epoch,learningrate)
    # model = NN3(datapath, batch_size, h1,h2,h3, epoch, learningrate)
    # train_csv = pa.read_csv(datapath)
    # test_data = np.array(train_csv[8000])
    # print 'Result:', model.predict(test_data).argmax()

#result:
# batch_size= 60
# hidden = 100
# h1=784
# h2 = 300
# h3 = 128
# epoch = 40
# learningrate = 0.01     93.855721393  %
#                0.02     94.5273631841 %
#                0.03     94.7512437811 %




