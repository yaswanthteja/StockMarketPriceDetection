from django.shortcuts import render
from django.conf import settings
# Create your views here.
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def user_machine_learning(request):
    from .utility import stock_predictions
    result = stock_predictions.start_process()
    #result = result.to_html
    return render(request, 'users/ml_results.html', {'results': result})


def user_future_prediction(request):
    from .utility.FuturePredections import FuturePredImpl
    obj = FuturePredImpl()
    rslt = obj.startFuturePrediction()
    import pandas as pd
    rslt = pd.DataFrame(rslt)
    rslt = rslt.to_html
    return render(request, 'users/futures.html', {'data': rslt})


def view_dataset(request):
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "AMZN.csv"
    df = pd.read_csv(path)
    df = df.to_html
    return render(request,'users/view_dataset.html', {'data':df})



