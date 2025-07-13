from django.urls import path
from .views import *

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('scrape/', ScrapeTriggerView.as_view(), name='dashboard_scrape'),
    path('scrape/status/', ScrapeTriggerView.as_view(), name='scrape_status'),
    path('reevaluate/', TriggerEvaluationView.as_view(), name='reevaluate'),
    path('reevaluate/status/', TriggerEvaluationView.as_view(), name='reevaluate_status'),
    path('company/<int:pk>/', CompanyDetailView.as_view(), name='company_detail'),
    path('company/<int:pk>/upload_pdf/', PDFUploadView.as_view(), name='upload_pdf'),
    path('company/create/', CompanyCreateUpdateView.as_view(), name='company_create'),
    path('company/<int:pk>/edit/', CompanyCreateUpdateView.as_view(), name='company_edit'),
    path('company/toggle-active/', toggle_company_active, name='toggle_company_active'),
    path('pdf/toggle-active/', toggle_pdf_active, name='toggle_pdf_active'),
    path('queries/create/', QueryCreateUpdateView.as_view(), name='query_create'),
    path('queries/<int:pk>/edit/', QueryCreateUpdateView.as_view(), name='query_edit'),
    path('queries/<int:pk>/toggle-active/', toggle_query_active, name='toggle_query_active'),
    path('llm-evaluate/', LLMRunEvaluationView.as_view(), name='llm_evaluate'),
]
