import mxnet as mx
import numpy as np
import pandas as pa


def predata(datapath,isTrain):
    '''
    delete unused number and change sex:male:0  female:1
    :param datapath:
    :param isTrain:
    :return:
    '''
    data = pa.read_csv(datapath).values
    if isTrain == False:
        data2 = np.delete(data, [0, 2, 7, 8, 9, 10], axis=1)
        for i in data2:
            if i[1] == 'male':
                i[1] = 0
            else:
                i[1] = 1
        return data2
    else:
        data2 = np.delete(data, [0,1, 3, 8, 9, 10, 11], axis=1)
        for i in data2:
            if i[1] == 'male':
                i[1] = 0
            else:
                i[1] = 1
            label = data[:,1]
        return data2,label

def myNN(train_path,test_path,net,batch_size,epochs,learningrate):

    #create network Symbol(just a graph no any space)
    # net = createNetWorkL3(10,20,10)
    #  initial the element in the network
    #  the initial is not only give the first weight and bias, we need tell the computer to
    #  allocate the memory for the element in the network
    #  so what i can do :  SimpleBind let mxnet give related memory for network

    #download data to mxnet.io.NDIter
    train_iter = loaddata2DataIter(train_path,batchsize=batch_size)
    test_data = predata(test_path,False)
    test_data = test_data[:400]
    # test_iter = loaddata2DataIter(te)
    test_iter = mx.io.NDArrayIter(test_data, None, batch_size=batch_size)
    test_output = []
    train_info = train_iter.provide_data+train_iter.provide_label
    input_info = dict(train_info)    #data:shape;softmax_label:shape

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
    exe_info = dict(zip(net.list_arguments(), exe.arg_arrays))

    #initial weight and bias
    init = mx.init.Uniform(scale=0.08)
    for i, j in exe_info.items():
        if i not in input_info:
            init(i, j)
    #combile network arguments and simple_bind allocate array
    exe_info = dict(zip(net.list_arguments(), exe.arg_arrays))
    # print exe_info
    # print 'provide len',train_iter.provide_data[0],train_iter.provide_label[0]
    # data = exe_info[train_iter.provide_data[0][0]]
    #
    # label = exe_info[train_iter.provide_label[0][0]]
    # print data,label
    # init = mx.init.Uniform(scale=0.08)
    # for i,j in exe_info.items():
    #     if i not in input_info:
    #         init(i,j)

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
            # print 'batch.data[0]',batch.data[0].asnumpy()
            # print 'exe_info[\'data\']',exe_info['data']
            exe_info['softmax_label'][:]= batch.label[0]
            # print 'batch labe',batch.label[0].asnumpy()
            # print 'batch label ',batch.label[0].asnumpy()
            exe.forward(is_train=True)
            exe.backward()
            #update
            for index, args in enumerate(zip(exe.arg_arrays,exe.grad_arrays)):
                weight, grad = args
                updater(index,grad,weight)#update each weight and grad

            metric.update(batch.label,exe.outputs)

            # pred_prob = exe.outputs[0].asnumpy()
            # pred = np.argmax(pred_prob, axis=1)
            # print pred
            # print 'output :', exe.outputs[0].asnumpy()
            # print 'compare label:', batch.label[0].asnumpy()
            num += 1
            if num % 20 == 0:
                print 'epoch',epoch,'iter',num,'metric',metric.get()
        # out = []
        # for catch in test_iter:
        #     # print 'test data', catch.data[0].asnumpy()
        #     exe_info['data'][:] =catch.data[0]
        #     exe.forward(is_train=False)
        #     print 'output :',exe.outputs[0].asnumpy()
        # # exe_info['data'][:] = mx.nd.array(test_data)
        # # exe.forward()






    # return test_output


def loaddata2DataIter(train_path,batchsize):

    '''
    :param datapath: all data path
    :param batchsize:
    :return: train_iter, test_iter
    '''

    train_d, train_l= predata(train_path,True)
    train_iter = mx.io.NDArrayIter(train_d, train_l, batch_size=batchsize)
    return train_iter



def update_SGD(learnigrate,train_iter):
    opt_SGD = mx.optimizer.SGD(
        learning_rate=learnigrate,
        momentum=0.1,
        wd=0.001,
        rescale_grad=1.0/(train_iter.batch_size )#True or False
    )
    update = mx.optimizer.get_updater(opt_SGD)
    return update
    # return opt_SGD


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
    act = mx.sym.Activation(data=fc, name='actlayer', act_type='tanh')
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
    act3 = mx.sym.Activation(data=fc3, name='act3', act_type='sigmoid')
    net = mx.sym.SoftmaxOutput(data=act3, name='softmax')
    return net



#def myNN(train_path,test_path,net,batch_size,epochs, learningrate):
if __name__ == '__main__':
    train_path = "/home/xuqian/mxnet2/data/titinic_train.csv"
    test_path = "/home/xuqian/mxnet2/data/titinic_test.csv"
    # t = predata(test_path,False)
    # print t.shape
    batch_size = 8
    hidden = 5
    h1 = 10
    h2 = 50
    h3 = 10
    epoch = 10
    learningrate = 2
    net1 = createNetWorkL1(hidden=hidden)
    net3 = createNetWorkL3(h1=h1,
                          h2=h2,
                          h3=h3
                          )
    myNN(
        train_path=train_path,
        test_path=test_path,
        net=net3,
        batch_size=batch_size,
        epochs=epoch,
        learningrate=learningrate
    )













