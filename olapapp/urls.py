from django.urls import path
from .views import olap_chart_view

urlpatterns = [
    path('olap-chart/', olap_chart_view, name='olap_chart'),
]
