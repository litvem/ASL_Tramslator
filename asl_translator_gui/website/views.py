from django.shortcuts import render

# Home view
def home(request):
    return render(request, "home.html", {})

# Login view
def login(request):
    return render(request, "login.html", {})

# Model training view
def training(request):
    return render(request, "training.html", {})