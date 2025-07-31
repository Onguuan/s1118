import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from skopt.space import Space
from skopt import gp_minimize
from skopt.space import Categorical
from tqdm import tqdm
import dataset
import SDSNN_model

# Create directories to save model parameters for different datasets
save_dir = "param_Fashion_MNIST"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir = "param_cifar10"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir = "param_cifar100"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def block(t1, t2, t3, t4, t5, t6, epochs=20):
    """
       Train and evaluate the model with given time step parameters.

       :param t1, t2, t3, t4, t5, t6: Time step parameters for the model.
       :param epochs: Number of training epochs.
       :return: Negative test accuracy (to be minimized by the optimizer).
    """
    print("t1, t2, t3, t4, t5, t6:", t1, t2, t3, t4, t5, t6)
    # Initialize lists to store training and testing metrics
    train_acc_list = [0 for i in range(epochs)]
    test_acc_list = [0 for i in range(epochs)]
    train_loss_list = [0 for i in range(epochs)]
    test_loss_list = [0 for i in range(epochs)]
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--type', default='Fashion_MNIST', help='Fashion_MNIST|cifar10|cifar100')
    parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed')
    parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to print')
    parser.add_argument('--test_interval', type=int, default=1,  help='the epochs to wait before another test')
    parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy one')
    parser.add_argument('--criterion', default='MSE', help='MSE|cross_entropy')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"args.type: {args.type}")
    # Load the dataset and initialize the model based on the dataset type
    assert args.type in ['Fashion_MNIST', 'cifar10', 'cifar100'], args.type
    if args.type == 'Fashion_MNIST':
        train_loader, test_loader = dataset.getfa(batch_size=args.batch_size)
        model = SDSNN_model.Fashion_MNIST(batch_size=args.batch_size, t1=t1, t2=t2, t3=t3, t4=t4)
    if args.type == 'cifar10':
        train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=0)
        model = SDSNN_model.cifar10(batch_size=args.batch_size, t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6)
    if args.type == 'cifar100':
        train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=0)
        model = SDSNN_model.cifar100(batch_size=args.batch_size, t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6)
    model = torch.nn.DataParallel(model, device_ids=range(1))
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 * args.epochs)

    train_correct_ave = 0
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for data, target in tqdm(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            if args.criterion == "MSE":
                target = F.one_hot(target, 10).float()
                criterion = torch.nn.MSELoss()
                loss = criterion(output, target)
                train_loss += loss.item()
            else:
                loss = F.cross_entropy(output, target)
                train_loss += loss.item()
                loss = loss.cuda()
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = pred.cpu().eq(indx_target).sum()
            train_correct += correct
        if epoch % args.test_interval == 0:
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / len(train_loader.dataset)
            train_correct_ave += train_acc
            print('\tTrain set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                epoch, train_loss, train_correct, len(train_loader.dataset), train_acc))
            train_loss_list[epoch] = train_loss
            train_acc_list[epoch] = train_acc

            # Evaluation on the test dataset
            with torch.no_grad():
                model.eval()
                test_loss = 0
                correct = 0
                for data, target in test_loader:
                    indx_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)
                    test_loss += F.cross_entropy(output, target).item()
                    pred = output.data.max(1)[1]
                    correct += pred.cpu().eq(indx_target).sum()
                test_loss = test_loss / len(test_loader)
                test_acc = 100. * correct / len(test_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                    test_loss, correct, len(test_loader.dataset), test_acc))
                test_loss_list[epoch] = test_loss
                test_acc_list[epoch] = test_acc
                if epochs != 0:
                    model_path = f"param_cifar10/SD_{t1, t2, t3, t4, t5, t6}_cifar10_bayesian_step_2.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved at epoch {epoch}, timestep {t1, t2, t3, t4, t5, t6} to {model_path}")
        scheduler.step()
    result = - test_acc
    print(result)

    return result.item()

def my_function():
    """
        Main function to perform Bayesian optimization.
    """
    start_time = time.time()
    results = []
    # stage 1
    for t1 in range(1, 6):
        result = -block(t1, t1, t1, t1, t1, t1, 40)
        print(result)
        results.append(result)

    best_result = max(results)
    print(best_result)
    # Filter out results that are less than 90% of the best result
    threshold = best_result * 0.9
    best_t1_values = [t1 for t1, result in zip(range(1, 6), results) if result >= threshold]
    print("best_t1_values:", best_t1_values)

    # the search space for Bayesian optimization
    search_space = Space([
        Categorical(best_t1_values, name="t1"),  # t1 values from best_t1_values
        Categorical(best_t1_values, name="t2"),
        Categorical(best_t1_values, name="t3"),
        Categorical(best_t1_values, name="t4"),
        Categorical(best_t1_values, name="t5"),
        Categorical(best_t1_values, name="t6"),
    ])

    # the objective function for Bayesian optimization
    def objective_function(params):
        t1, t2, t3, t4, t5, t6 = params
        return block(t1, t2, t3, t4, t5, t6)

    # Perform Bayesian optimization using gp_minimize
    optimizationr = gp_minimize(func=objective_function,
                              dimensions=search_space,
                              acq_func="gp_hedge",
                              n_calls=100,
                              random_state=1234,
                              verbose=True,
                              n_jobs=-1)

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total execution time：{total_time} seconds")

    print('Best Accuracy: %.3f' % (optimizationr.fun))
    print('Best Parameters: %s' % (optimizationr.x))


my_function()
