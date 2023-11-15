from django.shortcuts import render

# Home view
def home(request):
    return render(request, "home.html", {})