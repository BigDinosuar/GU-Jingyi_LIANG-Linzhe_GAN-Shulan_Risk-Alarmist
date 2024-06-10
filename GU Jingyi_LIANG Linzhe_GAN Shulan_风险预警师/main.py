import os
import json
import time
import math
import matplotlib.pyplot as plt
import datetime as dt
from core.data_prepare import DataLoader
from core.model import Model
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    start_time = time.time()
    
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['columns'],
        configs['data']['type']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    if configs['training']['use_generator']:
        # Out-of memory generative training
        total_samples = data.len_train - configs['data']['sequence_length']
        batch_size = configs['training']['batch_size']
        sample_rate = 0.5  # 只用一半的数据

        # 计算采样后的数据量
        sampled_data_size = int(total_samples * sample_rate)
        steps_per_epoch = math.ceil(sampled_data_size / batch_size)

        # 确保每个epoch都有不同的数据
        indices = np.random.permutation(total_samples)[:sampled_data_size]
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise'],
                indices=indices
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )
    else:
        # In-memory training
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        model.train(
            x,
            y,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            save_dir=configs['model']['save_dir']
        )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)
    np.save(f'predictions-{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}.npy', predictions)

    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test)
    
    print('mse: ', model.mse(y_test, predictions))
    print('rmse: ', model.rmse(y_test, predictions))
    print('mae: ', model.mae(y_test, predictions))

    print('Time taken: ', time.time() - start_time, ' seconds')


if __name__ == '__main__':
    main()
