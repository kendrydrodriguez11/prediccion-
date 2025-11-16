from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
from .services import StockPredictor

@csrf_exempt
def predict_stock(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No se proporcionó archivo'}, status=400)
            
            file_path = default_storage.save(f'tmp/{uploaded_file.name}', uploaded_file)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            params = json.loads(request.POST.get('params', '{}'))
            n_test = params.get('n_test', 300)
            epochs = params.get('epochs', 100)
            hidden_dim = params.get('hidden_dim', 16)
            n_layers = params.get('n_layers', 2)
            lr = params.get('lr', 0.01)
            
            data = StockPredictor.prepare_data(full_path, n_test)
            
            predictor = StockPredictor(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                lr=lr,
                epochs=epochs
            )
            
            predictions = predictor.train_and_predict(
                data['trainX'],
                data['trainy'],
                data['testX']
            )
            
            predicted_prices = StockPredictor.reconstruct_prices(
                predictions,
                data['close_prices'],
                n_test
            )
            
            actual_prices = data['close_prices'][-n_test:].tolist()
            dates = data['dates'][-n_test:].dt.strftime('%Y-%m-%d').tolist()
            
            metrics = StockPredictor.calculate_metrics(actual_prices, predicted_prices)
            
            default_storage.delete(file_path)
            
            return JsonResponse({
                'success': True,
                'dates': dates,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices,
                'metrics': metrics
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'predictor/index.html')

def index(request):
    return render(request, 'predictor/index.html')