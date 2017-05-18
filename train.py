import numpy as np
from math import exp, log, sqrt
import pickle, gzip, argparse

np.random.seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument("--momentum")
parser.add_argument("--batch_size")
parser.add_argument("--activation")
parser.add_argument("--num_hidden")
parser.add_argument("--anneal")
parser.add_argument("--lr")
parser.add_argument("--opt")
parser.add_argument("--mnist")
parser.add_argument("--expt_dir")
parser.add_argument("--loss")
parser.add_argument("--save_dir")
parser.add_argument("--sizes")
args = parser.parse_args()

lr = float(args.lr)

test_temp = args.anneal
if test_temp == 'true':
    if_anneal = True
else:
    if_anneal = False

momentum = float(args.momentum)

batch_size = int(args.batch_size)

max_epochs = 10

test_temp = args.activation
if test_temp == 'tanh':
    act_fun = 1
else:
    act_fun = 0

num_hidden = int(args.num_hidden)

layer_vals = np.zeros(num_hidden + 2)

layer_vals[0] = 784

layer_vals[1] = 10

test_temp = args.opt
if test_temp == 'gd':
    algorithm = 0
elif test_temp == 'momentum':
    algorithm = 1
elif test_temp == 'nag':
    algorithm = 2
elif test_temp == 'adam':
    algorithm = 3

path_to_pickled_sets = args.mnist

log_files_directory = args.expt_dir

expected_files_directory = args.expt_dir

test_temp = args.loss
if test_temp == 'sq':
    loss_fun = 0
if test_temp == 'ce':
    loss_fun = 1

save_directory = args.save_dir

sizes_str = args.sizes
sizes_list = sizes_str.strip().split(',')

for e12 in range(1, num_hidden + 1):
    layer_vals[e12] = sizes_list[e12 - 1]

original_lr = lr
layer_vals[num_hidden + 1] = 10
layer_vals = layer_vals.astype(int)
layer_maxval = np.amax(layer_vals)
nn_activation = np.random.rand(num_hidden + 2, int(layer_maxval + 1))
nn_derivative_cost_wrt_s = np.random.rand(num_hidden + 2, int(layer_maxval + 1))
num_layers = int(num_hidden + 2)
nn_weights = np.random.uniform(-0.999, 0.999, (num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_gradients = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_activation[0][784] = 1
nn_updates = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_weights_look_ahead = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_weights_previous_iteration = np.array(nn_weights)
nn_adam_mt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_adam_vt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))

nn_adam_mt.fill(lr)
nn_adam_vt.fill(lr)
beta1 = 0.9
beta2 = 0.999
eps = 10 ** -8

for i in range(0, num_layers):
    nn_activation[i][int(layer_vals[i])] = 1

def function(s):

    if act_fun == 0:

        if s > 300:
            return 1

        if s < -300:
            return 0

        return 1 / (1 + exp(-s))

    elif act_fun == 1:

        if s > 300:
            return 1

        if s < -300:
            return -1

        return (exp(s) - exp(-s)) / (exp(s) + exp(-s))


def sigmoid_function(s):

    if s > 300:
        return 1

    if s < -300:
        return 0

    return 1 / (1 + exp(-s))


def predict(pic_bits):

    for i in range(0, 784):
        nn_activation[0][i] = pic_bits[i]

    for i in range(1, int(num_layers-1)):
        for j in range(0, int(layer_vals[i])):
            x = 0
            for k in range(0, int(layer_vals[i - 1]) + 1):
                x += nn_weights[i][k][j] * nn_activation[i - 1][k]

            nn_activation[i][j] = function(x)

    for i in range(0,10):
        x = 0
        for j in range(0, layer_vals[num_layers - 2]):
            x += nn_activation[num_layers - 2][j] * nn_weights[num_layers - 1][j][i]

        nn_activation[num_layers - 1][i] = sigmoid_function(x)

    mx = 0
    mxindex = 0

    for i in range(0, 10):
        if nn_activation[num_layers - 1][i] > mx:
            mx = nn_activation[num_layers - 1][i]
            mxindex = i

    return mxindex


def derivative_activ_fun_wrt_s_final(s):
    return function(s) - function(s) ** 2


def derivative_activ_fun_wrt_s(s):

    if act_fun == 0:
        return function(s) - function(s) ** 2

    if act_fun == 1:
        return 1 - (function(s)) ** 2


def derivative_cost_wrt_x_final(a7, out_vector):
        return nn_activation[num_layers - 1][a7] - out_vector[a7]


def gradient_descent():

    global nn_weights, nn_gradients

    for a14 in range(1, num_layers):
        for a15 in range(0, layer_vals[a14]):
            for a16 in range(0, layer_vals[a14 - 1] + 1):
                nn_weights[a14][a16][a15] -= lr * nn_gradients[a14][a16][a15]


def momentum_gradient_descent():

    global nn_weights, nn_updates

    for a14 in range(1, num_layers):
        for a15 in range(0, layer_vals[a14]):
            for a16 in range(0, layer_vals[a14 - 1] + 1):
                nn_updates[a14][a16][a15] = momentum * nn_updates[a14][a16][a15] + lr * nn_gradients[a14][a16][a15]
                nn_weights[a14][a16][a15] -= nn_updates[a14][a16][a15]


def nag():

    global nn_weights, nn_updates, nn_gradients

    for a14 in range(1, num_layers):
        for a15 in range(0, layer_vals[a14]):
            for a16 in range(0, layer_vals[a14 - 1] + 1):
                nn_updates[a14][a16][a15] = momentum * nn_updates[a14][a16][a15] + lr * nn_gradients[a14][a16][a15]
                nn_weights[a14][a16][a15] -= nn_updates[a14][a16][a15]


def nesterov_find_gradients():

    global nn_weights, nn_weights_look_ahead, nn_derivative_cost_wrt_s, nn_gradients, nn_activation, momentum, nn_updates, nn_activation

    for a14 in range(1, num_layers):
        for a15 in range(0, layer_vals[a14]):
            for a16 in range(0, layer_vals[a14 - 1] + 1):
                nn_weights_look_ahead[a14][a16][a15] = nn_weights[a14][a16][a15] - momentum * nn_updates[a14][a16][a15]

    for a8 in range(num_layers - 2, 0, -1):
        for a9 in range(0, layer_vals[a8] + 1):
            x = 0
            for a10 in range(0, layer_vals[a8 + 1] + 1):

                if a8 + 1 == num_layers - 1 and a10 == 10:
                    continue

                x += nn_derivative_cost_wrt_s[a8 + 1][a10] * nn_weights_look_ahead[a8 + 1][a9][a10]

            x *= derivative_activ_fun_wrt_s(nn_activation[a8][a9])
            nn_derivative_cost_wrt_s[a8][a9] = x

    for a11 in range(1, num_layers):
        for a12 in range(0, layer_vals[a11]):
            for a13 in range(0, layer_vals[a11 - 1] + 1):
                nn_gradients[a11][a13][a12] += nn_activation[a11 - 1][a13] * nn_derivative_cost_wrt_s[a11][a12]


def adam():

    global nn_weights, nn_gradients, nn_adam_mt, nn_adam_vt, eps, beta1, beta2

    for a14 in range(1, num_layers):
        for a15 in range(0, layer_vals[a14]):
            for a16 in range(0, layer_vals[a14 - 1] + 1):

                nn_weights[a14][a16][a15] -= (lr / sqrt(nn_adam_vt[a14][a16][a15] + eps)) * nn_adam_mt[a14][a16][a15]

                nn_adam_mt[a14][a16][a15] = beta1 * nn_adam_mt[a14][a16][a15] + (1 - beta1) * nn_gradients[a14][a16][a15]
                nn_adam_vt[a14][a16][a15] = beta2 * nn_adam_vt[a14][a16][a15] + (1 - beta2) * (nn_gradients[a14][a16][a15]) ** 2


def predict_loss(X, Y):

    net_loss = 0

    for i in range(0, 784):
        nn_activation[0][i] = X[i]

    for i in range(1, int(num_layers-1)):
        for j in range(0, int(layer_vals[i])):
            x = 0
            for k in range(0, int(layer_vals[i - 1]) + 1):
                x += nn_weights[i][k][j] * nn_activation[i - 1][k]

            nn_activation[i][j] = function(x)

    for i in range(0,10):
        x = 0
        for j in range(0, layer_vals[num_layers - 2]):
            x += nn_activation[num_layers - 2][j] * nn_weights[num_layers - 1][j][i]

        nn_activation[num_layers - 1][i] = sigmoid_function(x)

    if loss_fun == 0:
        for a17 in range(0, 10):
            net_loss += (Y[a17] - nn_activation[num_layers - 1][a17]) ** 2

    elif loss_fun == 1:
        for a17 in range(0, 10):

            yy = Y[a17]
            aa = nn_activation[num_layers - 1][a17]

            if aa == 0:
                if yy != 0:
                    net_loss += 10 ** 100

            elif aa == 1:
                if yy != 1:
                    net_loss += 10 ** 100
            else:
                net_loss += -(yy * log(aa) + (1 - yy) * log(1 - aa))

    return net_loss


def validation_loss(Xvalid, Yvalid):

    num_testcases = Yvalid.shape[0]
    total_loss = 0

    for a1 in range(0, num_testcases):
        loss = predict_loss(Xvalid[a1], Yvalid[a1])

        total_loss += loss

    return total_loss / num_testcases


def train(X, Y, Xvalid, Yvalid, path_to_loss_logs, path_to_error_logs):

    global nn_weights, nn_gradients, nn_updates, nn_activation, nn_adam_vt, nn_adam_mt, nn_derivative_cost_wrt_s
    global nn_weights_look_ahead, nn_weights_previous_iteration, lr

    net_loss = 0
    previous_validation_error = 10 ** 150
    hundred_loss = 0
    hundred_num_correct = 0

    a1 = 0

    while a1 < 7:
        num_correct = 0

        nn_weights_previous_iteration = np.array(nn_weights)

        nn_updates.fill(0)

        num_points_seen = 0

        for a2 in range(0, X.shape[0]):

            prediction = predict(X[a2])

            correct_prediction = -1

            for a22 in range(0,10):
                if Y[a2][a22] == 1:
                    correct_prediction = a22
                    break

            if prediction == correct_prediction:
                num_correct += 1

            nn_activation.fill(0)

            for i in range(0, num_layers):
                nn_activation[i][int(layer_vals[i])] = 1

            for a3 in range(0, 784):
                nn_activation[0][a3] = X[a2][a3]

            for a4 in range(1, num_layers - 1):
                for a5 in range(0, layer_vals[a4]):
                    x = 0
                    for a6 in range(0, layer_vals[a4 - 1] + 1):
                        x += nn_activation[a4 - 1][a6] * nn_weights[a4][a6][a5]

                    nn_activation[a4][a5] = function(x)

            for a20 in range(0,10):
                x = 0
                for a21 in range(0, layer_vals[num_layers - 2]):
                    x += nn_activation[num_layers - 2][a21] * nn_weights[num_layers - 1][a21][a20]

                nn_activation[num_layers - 1][a20] = sigmoid_function(x)

            for a7 in range(0, 10):
                if loss_fun == 0:
                    nn_derivative_cost_wrt_s[num_layers - 1][a7] = derivative_cost_wrt_x_final(a7, Y[
                        a2]) * derivative_activ_fun_wrt_s_final(nn_activation[num_layers - 1][a7])

                elif loss_fun == 1:
                    nn_derivative_cost_wrt_s[num_layers - 1][a7] = nn_activation[num_layers - 1][a7] - Y[a2][a7]

            if algorithm != 2:

                for a8 in range(num_layers - 2, 0, -1):
                    for a9 in range(0, layer_vals[a8] + 1):
                        x = 0
                        for a10 in range(0, layer_vals[a8 + 1] + 1):

                            if a8 + 1 == num_layers - 1 and a10 == 10:
                                continue

                            x += nn_derivative_cost_wrt_s[a8 + 1][a10] * nn_weights[a8 + 1][a9][a10]

                        x *= derivative_activ_fun_wrt_s(nn_activation[a8][a9])
                        nn_derivative_cost_wrt_s[a8][a9] = x

                for a11 in range(1, num_layers):
                    for a12 in range(0, layer_vals[a11]):
                        for a13 in range(0, layer_vals[a11 - 1] + 1):
                            nn_gradients[a11][a13][a12] += nn_activation[a11 - 1][a13] * nn_derivative_cost_wrt_s[a11][a12]


            else:
                nesterov_find_gradients()

            num_points_seen += 1

            if loss_fun == 0:
                for a17 in range(0,10):
                    net_loss += (Y[a2][a17] - nn_activation[num_layers - 1][a17])**2

            elif loss_fun == 1:
                for a17 in range(0,10):

                    yy = Y[a2][a17]
                    aa = nn_activation[num_layers - 1][a17]

                    if aa == 0:
                        if yy != 0:
                            net_loss += 10 ** 100

                    elif aa == 1:
                        if yy != 1:
                            net_loss += 10 ** 100
                    else:
                        net_loss += -(yy * log(aa) + (1 - yy) * log(1 - aa))

            if num_points_seen % batch_size == 0:

                if algorithm == 0:
                    gradient_descent()

                elif algorithm == 1:
                    momentum_gradient_descent()

                elif algorithm == 2:
                    nag()

                elif algorithm == 3:
                    adam()

                nn_gradients.fill(0)

                #print("Epoch " + str(a1) + ", Step " + str(num_points_seen) + ", Loss: " + str(
                #    net_loss / batch_size) + ", lr: " + str(lr))

                hundred_loss += net_loss
                hundred_num_correct += num_correct

                if (num_points_seen/batch_size) % 100 == 0:
                    f = open(path_to_loss_logs, 'a')
                    f.write(str("Epoch " + str(a1) + ", Step " + str(int(num_points_seen/batch_size)) + ", Loss: " + str(hundred_loss/(batch_size*100)) + ", lr: " + str(lr) + '\n'))
                    f.close()

                    f = open(path_to_error_logs, 'a')
                    f.write(str("Epoch " + str(a1) + ", Step " + str(
                        int(num_points_seen / batch_size)) + ", Error: " + format(
                        (1 - hundred_num_correct / (batch_size * 100)) * 100, '.2f') + ", lr: " + str(lr) + '\n'))

                    hundred_loss = 0
                    hundred_num_correct = 0

                net_loss = 0
                num_correct = 0

        if if_anneal:

            current_iteration_validation_error = validation_loss(Xvalid, Yvalid)

            if current_iteration_validation_error > previous_validation_error:
                a1 -= 1
                nn_weights = nn_weights_previous_iteration
                lr /= 2

            previous_validation_error = current_iteration_validation_error

        a1 += 1

def test(X, Y, path_to_predict_file):
    num_testcases = Y.shape[0]
    num_correct = 0

    for a1 in range(0, num_testcases):
        prediction = predict(X[a1])

        correct_prediction = -1

        for a2 in range(0, 10):
            if Y[a1][a2] == 1:
                correct_prediction = a2
                break

        if prediction == correct_prediction:
            num_correct += 1

        f = open(path_to_predict_file, 'a')
        f.write(str(str(prediction) + '\n'))
        f.close()

    #print("Accuracy is " + str(num_correct / num_testcases))


def master():

    global nn_weights, nn_updates, nn_weights_look_ahead, nn_adam_mt, nn_adam_vt, lr, if_anneal

    f = gzip.open(path_to_pickled_sets, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    X = train_set[0]
    Y = train_set[1]

    Ynew = np.zeros((Y.size, 10))

    for i in range(0, Y.size):
        Ynew[i][int(Y[i])] = 1

    Y = Ynew

    Xvalid = valid_set[0]
    Yvalid = valid_set[1]

    Ynew_valid = np.zeros((Yvalid.size, 10))

    for i in range(0, Yvalid.size):
        Ynew_valid[i][int(Yvalid[i])] = 1

    Yvalid = Ynew_valid

    path_to_loss_logs = log_files_directory + '/log_loss_train.txt'

    path_to_error_logs = log_files_directory + '/log_error_train.txt'

    train(X, Y, Xvalid, Yvalid, path_to_loss_logs, path_to_error_logs)

    with open(str(save_directory + 'weights_and_biases.pkl'), 'wb') as save_file:
        pickle.dump(nn_weights_look_ahead, save_file, -1)

    Xtest = test_set[0]
    Ytest = test_set[1]

    Ynew_test = np.zeros((Ytest.size, 10))

    for i in range(0, Ytest.size):
        Ynew_test[i][int(Ytest[i])] = 1

    Ytest = Ynew_test

    path_to_expected_file = expected_files_directory + '/test_predictions.txt'

    test(Xtest, Ytest, path_to_expected_file)

    path_to_expected_file = expected_files_directory + '/valid_predictions.txt'

    test(Xvalid, Yvalid, path_to_expected_file)

    nn_weights = nn_weights = np.random.uniform(-0.999, 0.999, (num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_updates = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_weights_look_ahead = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_mt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_vt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_mt.fill(lr)
    nn_adam_vt.fill(lr)
    lr = original_lr

    if_anneal = False

    path_to_loss_logs = log_files_directory + '/log_loss_valid.txt'

    path_to_error_logs = log_files_directory + '/log_error_valid.txt'

    train(Xvalid, Yvalid, Xtest, Ytest, path_to_loss_logs, path_to_error_logs)

    nn_weights = nn_weights = np.random.uniform(-0.999, 0.999,
                                                (num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_updates = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_weights_look_ahead = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_mt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_vt = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
    nn_adam_mt.fill(lr)
    nn_adam_vt.fill(lr)
    lr = original_lr

    path_to_loss_logs = log_files_directory + '/log_loss_test.txt'

    path_to_error_logs = log_files_directory + '/log_error_test.txt'

    if_anneal = False

    train(Xtest, Ytest, Xtest, Ytest, path_to_loss_logs, path_to_error_logs)

master()
