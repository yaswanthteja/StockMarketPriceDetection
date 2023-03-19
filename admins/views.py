from django.shortcuts import render, HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel


# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})


def admin_forecasting(request):
    from .utility.foreCast_Model import AmazonFuturePrediction
    object = AmazonFuturePrediction()
    prediction  = object.start_future_prediction()
    import pandas as pd
    df = pd.DataFrame(prediction)
    print(df)
    df = df.to_html

    return render(request, 'admins/admin_future.html',{'data': df})
