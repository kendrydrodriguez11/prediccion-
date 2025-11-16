import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .ml_models.mamba import StockPredictorNet

class StockPredictor:
    def __init__(self, hidden_dim=16, n_layers=2, lr=0.01, wd=1e-5, epochs=100, use_cuda=False):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.model = None
    
    def train_and_predict(self, trainX, trainy, testX):
        self.model = StockPredictorNet(len(trainX[0]), 1, self.hidden_dim, self.n_layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        xt = torch.from_numpy(trainX).float().unsqueeze(0)
        xv = torch.from_numpy(testX).float().unsqueeze(0)
        yt = torch.from_numpy(trainy).float()
        
        if self.use_cuda:
            self.model = self.model.cuda()
            xt = xt.cuda()
            xv = xv.cuda()
            yt = yt.cuda()
        
        for e in range(self.epochs):
            self.model.train()
            z = self.model(xt)
            loss = F.mse_loss(z, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if e % 10 == 0 and e != 0:
                print(f'Epoch {e} | Loss: {loss.item():.4f}')
        
        self.model.eval()
        with torch.no_grad():
            mat = self.model(xv)
            if self.use_cuda:
                mat = mat.cpu()
            yhat = mat.numpy().flatten()
        
        return yhat
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    @staticmethod
    def prepare_data(csv_path, n_test=300):
        data = pd.read_csv(csv_path)
        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        
        close = data['close'].values
        ratechg = data['pct_chg'].apply(lambda x: 0.01 * x).values
        
        data.drop(columns=['pre_close', 'change', 'pct_chg'], inplace=True)
        dat = data.iloc[:, 2:].values
        
        trainX = dat[:-n_test, :]
        testX = dat[-n_test:, :]
        trainy = ratechg[:-n_test]
        
        return {
            'trainX': trainX,
            'trainy': trainy,
            'testX': testX,
            'close_prices': close,
            'dates': data['trade_date'],
            'n_test': n_test
        }
    
    @staticmethod
    def reconstruct_prices(predictions, close_prices, n_test):
        predicted_prices = []
        for i in range(n_test):
            pred_price = close_prices[-n_test-1+i] * (1 + predictions[i])
            predicted_prices.append(pred_price)
        return predicted_prices