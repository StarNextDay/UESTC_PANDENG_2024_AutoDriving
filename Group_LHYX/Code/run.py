import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append('/media/panda/Elements SE/Emotake_wyb/Time-Series-Library-main')
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import pandas as pd
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter

# from utils.print_args import print_args

from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import StratifiedKFold  # 分层k折交叉验证

from layers.OurAttention import *
from layers.Model import *
from model.iTransformer import Model as iTransformer
import random
from typing import Callable, Optional, Dict
import datetime


class MultiTaskCrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: Dict, y: torch.Tensor, temperature, task_weights):
        assert len(x) == y.size(-1)

        return sum([self.ce(x[i]*temperature.exp(), y[:, i].long())*task_weights[i] for i in range(len(x))]) / sum(task_weights)


def get_args():
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet, iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, default='Ours data', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=300, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # classification task
    parser.add_argument('--num_class', type=int, default=4, help='[quality:3, ra:3, readiness:2, emotion:4]')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=True, action='store_true',
                        help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # our arguments
    parser.add_argument('--tasks', type=list, default=[3, 3, 2, 4], help='numbers of tasks')
    parser.add_argument('--depth', type=int, default=2, help='model depth')
    parser.add_argument('--epochs', type=int, default=300, help='training expochs')
    parser.add_argument('--dim', type=int, default=128, help='dimension')
    parser.add_argument('--eval_save_frq', type=int, default=10, help='save model frequency')
    parser.add_argument('--model_name', type=str, default="OurModel", help='model name')
    return parser.parse_args()


def get_data():
    # ? data
    au_data = np.load('././DATA/aus_delete_1_realva.npy')  # 195*300*35
    em_data = np.load('././DATA/ems_delete_1_realva.npy')  # 195*300*288
    hp_data = np.load('././DATA/hps_delete_1_realva.npy')  # 195*300*6

    n, s = au_data.shape[:2]
    hr_data = np.load('././DATA/hrs_delete_1_realva.npy')
    bp_data = np.load('././DATA/bps_delete_1_realva.npy').reshape(n, s, -1)

    # hr里面有nan值,单独处理
    hr_df = pd.DataFrame(hr_data)
    hr_df.fillna(0, inplace=True)
    hr_data = np.array(hr_df).reshape(n, s, -1)

    combined_data = np.concatenate((au_data, em_data, hp_data, hr_data, bp_data), axis=-1)
    # person_num * seq_num * seq_len(300)
    mean = np.mean(combined_data, axis=-1).reshape(n, s, -1)
    std = np.std(combined_data, axis=-1).reshape(n, s, -1)
    combined_data = (combined_data - mean) / std
    # combined_data = combined_data.transpose(0,2,1)
    # ! 暂时不转置

    # ? label
    va_df = pd.read_excel('././DATA/real_va.xlsx')
    va_df = va_df[(va_df["real v"] != 5) & (va_df["real a"] != 5)][["real v", "real a"]]
    va_data = np.array(va_df) - 5
    va_emo_mapping = {(True, True): "happy",
                      (True, False): "relief",
                      (False, True): "fear",
                      (False, False): "bored", }  # bool(v>0),bool(a>0)
    emo_label_mapping = {"happy": 0,
                         "relief": 1,
                         "fear": 2,
                         "bored": 3, }
    global real_label
    real_label = np.array([emo_label_mapping[va_emo_mapping[(x[0] > 0, x[1] > 0)]] for x in va_data])

    quality_lable = np.load('././DATA/quality_lable_delete_1_ml_feature_real_va.npy', allow_pickle=True)
    quality_lable = np.array(quality_lable, dtype=int)
    ra_lable = np.load('././DATA/ra_lable_delete_1_ml_feature_real_va.npy', allow_pickle=True)
    ra_lable = np.array(ra_lable, dtype=int)
    readiness_lable = np.load('././DATA/readiness_lable_delete_1_ml_feature_real_va.npy', allow_pickle=True)
    readiness_lable = np.array(readiness_lable, dtype=int)

    combined_label = np.stack([quality_lable, ra_lable, readiness_lable, real_label], axis=-1)

    # [3, 3, 2, 4]

    # if pred_type == 0:
    #     combined_label = quality_lable
    # elif pred_type == 1:
    #     combined_label = ra_lable
    # else:
    #     combined_label = readiness_lable

    task_weight = {i: torch.tensor(combined_label)[:, i].bincount() for i in range(args.num_class)}

    # ! resample
    new_data = []
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN

    n_samples, height, width = combined_data.shape
    data_reshaped = combined_data.reshape(n_samples, -1)

    def multi_label_to_single_label(labels):
        # Assuming labels are in the form of integers for simplicity
        factor = np.array([64, 16, 4, 1])  # 根据各标签类别的数目计算因子
        single_label = np.dot(labels, factor)
        return single_label

    single_label = multi_label_to_single_label(combined_label)

    ros = RandomOverSampler(random_state=42)
    data_resampled, single_label_resampled = ros.fit_resample(data_reshaped, single_label)

    def single_label_to_multi_label(single_label):
        inverse_factor = np.array([64, 16, 4, 1])
        labels = np.zeros((single_label.size, 4), dtype=int)
        for i in range(4):
            labels[:, i] = single_label // inverse_factor[i]
            single_label %= inverse_factor[i]
        return labels

    labels_resampled = single_label_to_multi_label(single_label_resampled)

    n_resampled = data_resampled.shape[0]
    data_resampled = data_resampled.reshape(n_resampled, height, width)

    # additive gaussian noise
    data_resampled += 0.01 * np.random.randn(*data_resampled.shape)

    return data_resampled, labels_resampled


def emotion_acc_f1(model_pred):
    total = len(test_index)
    pred = model_pred.cpu()

    y_test_ = y_test[:, -1].cpu()

    # 分别计算每一情绪准确率
    lable_emo = y_test_.tolist()
    # for i in test_index:
    #     lable_emo.append(real_label[i])
    # print(f'Emotion for each lable: {lable_emo}')

    emo_correct = [0 for i in range(4)]  # 4个相限的情绪
    emo_num = [0 for i in range(4)]
    for i in lable_emo:
        emo_num[i] += 1
    print(f'Num for each emotion: {emo_num}')
    for i in range(total):
        if pred[i] == y_test_[i]:
            emo_correct[lable_emo[i]] += 1
    emo_acc = [emo_correct[i] / emo_num[i] for i in range(4)]
    print(f'Accuracy for each emotion: {emo_acc}')

    # 四种情绪分别的f1 score
    emo_f1 = []
    emo_lable = []
    emo_pred = []
    for i in range(4):
        emo_lable = []
        emo_pred = []
        for j in range(total):
            if lable_emo[j] == i:
                emo_lable.append(y_test_[j])
                emo_pred.append(pred[j])
        emo_f1.append(f1_score(emo_lable, emo_pred, average='weighted'))
    print(f'F1 score for each type: {emo_f1}\n')


if __name__ == '__main__':
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    args = get_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = False  # !

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args) #!

    # 0:quality 1:ra 2:readiness

    if args.task_name == 'classification':

        combined_data, combined_label = get_data()
        # 193, 354, 300
        

        # combined_data = combined_data.transpose(0, 2, 1) # person_num*seq_num*seq_len(300) => person_num*seq_len(300)*seq_num

        args.seq_len = combined_data.shape[1]
        args.enc_in = combined_data.shape[2]
        # args.num_class = sum([3, 3, 2, 4])

        # [Baseline_vanilla_mlp, Baseline_concat_mlp, Ablation, OurModel]
        for Model in [eval(args.model_name)]:
            print("\n", str(Model))
            avg_acc = []
            avg_f1 = []
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            for i_fold, (train_index, test_index) in enumerate(kf.split(combined_data, combined_label[:, 0])):

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = Model(args).to(device=device)
                print(model)

                # model.load_state_dict(torch.load("./weights_20240524/epoch[22]_acc[0.756, 0.756, 0.837, 0.641].pth"))
                
                temperature = nn.Parameter(torch.tensor(1.0))
                optimizer = optim.Adam([{"params": model.parameters(), "lr": args.learning_rate},
                                        {"params": temperature, "lr": 100 * args.learning_rate}])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.9)
                # criterion = nn.CrossEntropyLoss() #!
                criterion = MultiTaskCrossEntropyLoss()

                X_train = torch.tensor(np.array(combined_data)[train_index], dtype=torch.float32).to(device)
                y_train = torch.tensor(np.array(combined_label)[train_index]).to(device)

                X_test = torch.tensor(np.array(combined_data)[test_index], dtype=torch.float32).to(device)
                y_test = torch.tensor(np.array(combined_label)[test_index]).to(device)

                task_weight = {i: torch.tensor(combined_label)[:, i].bincount() for i in range(args.num_class)}

                # train
                # loss_file_path = './results/loss_{}'.format(pred_type)
                loss_ = []
                max_acc = 0

                for epoch in range(args.epochs):
                    model.train()
                    optimizer.zero_grad()

                    sub_index = torch.arange(train_index.shape[0])
                    # sub_index = torch.randperm(train_index.shape[0])

                    grad_accu = 0
                    for i, batch_index in enumerate(torch.split(sub_index, args.batch_size, dim=0)):
                        train_data = X_train[batch_index]
                        train_lable = y_train[batch_index]

                        train_outputs = model(train_data)  # LSTM

                        # Compute loss
                        loss = criterion(train_outputs, train_lable, temperature, [1,1,1,1])
                        loss_.append(loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                        # print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t',
                        #     f'Fold [{i_fold}]',
                        #     f'Epoch [{str(epoch + 1).zfill(3)}/{args.epochs}]',
                        #     f'Batch [{str(i).zfill(2)}]\t',
                        #     f'Loss: {loss.item():.4f}',)

                    scheduler.step()

                    if (epoch+1) % args.eval_save_frq != 0:
                        continue

                    model.eval()

                    with torch.no_grad():
                        val_data = X_test.to(device)
                        val_lable = y_test.to(device)
                        val_outputs = model(val_data)

                    # _, predicted = torch.max(val_outputs, 1)
                    predicted = torch.stack([output.argmax(-1) for output in val_outputs.values()], dim=-1)

                    acc_sum = []
                    for i in range(val_lable.size(-1)):
                        correct = predicted[:, i].eq(val_lable[:, i].data).sum()
                        # total = len(val_lable) #!
                        total = val_lable[:, i].numel()
                        accuracy = correct / total
                        acc_sum.append(round(accuracy.item(),3))
                        # print(f'Task_{i} Accuracy: {accuracy * 100:.2f}%')

                        f1 = f1_score(y_test[:, i].cpu(), predicted[:, i].cpu(), average='weighted')
                        # print(f'Task_{i} F1-Score: {f1 * 100:.2f}%')

                    # emotion_acc_f1(predicted[:, -1])

                    avg_acc.append(accuracy.cpu())
                    avg_f1.append(f1)

                    # path = f"./weights/fold_{i_fold}/"
                    # ckpt = path + f"epoch[{epoch+1}]_acc{acc_sum}.pth"
                    # print(f"save to {ckpt}\n")
                    # os.makedirs(f"./weights/fold_{i_fold}", exist_ok=True)
                    # torch.save(model.state_dict(), ckpt)
                    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t',
                          f'Model [{str(Model)}]',
                          f'Fold [{i_fold}]',
                          f'Epoch [{str(epoch + 1).zfill(3)}/{args.epochs}]',
                          f'Loss: {loss.item():.4f}',
                          f'Acc: {acc_sum}')

                print("mean acc:", np.array(avg_acc).mean())
                print("mean f1:", np.array(avg_f1).mean())

                # from torchsummary import summary
                # summary(model)
                # np.save('./loss', loss_)



