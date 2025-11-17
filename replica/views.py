from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')

def chatbot(request):
    return render(request, 'chatbot.html')

def prediction(request):
    return render(request, 'prediction.html')