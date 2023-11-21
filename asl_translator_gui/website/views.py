from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout

# Home
def home(request):
    return render(request, "home.html", {})

# About
def about(request):
    return render(request, "about.html", {})

# Login
def login_user(request):
    # If the form was filled in and button clicked
    if request.method == "POST":
        # Get input from login form
        username = request.POST['username']
        password = request.POST['password']
        # Use Django authenticate system to pass in username and password from the form
        user = authenticate(request, username=username, password=password)
        # Login if the form was filled in
        if user is not None:
            login(request, user)
            return redirect('training')
        # Redirect if login was not successful 
        else:
            return redirect('login')
    # If the form was not filled in, show the login page    
    else:
        return render(request, "login.html", {})

# Logout
def logout_user(request):
    logout(request)
    return redirect('home')

# Model training
def training(request):
    return render(request, "training.html", {})