from api.models import CompanyProfile, Query, EvaluationResult, PDFFile, PDFScrapeDate
from .forms import CompanyProfileForm, CompanyURLFormSet, QueryForm
from ..backend.llm_module.llm_db_interface import LLMDBInterface

import json

import hashlib
import threading
import traceback

from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.generic import TemplateView, DetailView
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse, reverse_lazy
from django.utils.timezone import now
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Max, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.core.cache import cache
from django.core.management import call_command
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from .serializers import LLMQuerySerializer, LLMResultSerializer
from backend.llm_module.evaluator import LLMEvaluator
from backend.llm_module.llm_provider import HuggingFaceLLMProvider
from ..backend.llm_module.llm_db_interface import LLMDBInterface






SCRAPING_STATUS_KEY = "scraping_running"
EVALUATION_STATUS_KEY = "evaluation_running"


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "api/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        search_query = self.request.GET.get("q", "")
        companies = CompanyProfile.objects.all()

        if search_query:
            companies = companies.filter(
                Q(name__icontains=search_query) | Q(industry__icontains=search_query)
            )

        companies = companies.annotate(
            pdf_count=Count("pdfs", distinct=True),
            last_evaluated=Max("evaluationresult__timestamp"),
        )


        queries = Query.objects.all()

        # Auswertung: sind alle aktiv?
        all_companies_active = not companies.exclude(active=True).exists()
        all_queries_active = not queries.exclude(active=True).exists()

        # Build evaluation matrix: latest result per (company, query)
        eval_matrix = {}
        for company in companies:
            eval_matrix[company.id] = {}
            for query in queries:
                result = EvaluationResult.objects.filter(
                    company=company, query=query
                ).order_by("-timestamp").first()
                eval_matrix[company.id][query.id] = result

        context.update({
            "companies": companies,
            "queries": queries,
            "eval_matrix": eval_matrix,
            "search_query": search_query,
            "all_companies_active": all_companies_active,
            "all_queries_active": all_queries_active,
            "scraping_running": cache.get("scraping_running", False)
        })
        return context


class AnalyzeView(View):
    def get(self, request):
        return HttpResponse("Analyze something here.")
    
class CompanyDetailView(DetailView):
    model = CompanyProfile
    template_name = 'api/company_detail.html'
    context_object_name = 'company'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        company = self.get_object()
        context['scraped_pdfs'] = company.pdfs.filter(source='webscraped')
        context['manual_pdfs'] = company.pdfs.filter(source='manual')
        context['urls'] = company.urls.all()
        return context


class PDFUploadView(View):
    def post(self, request, pk):
        company = get_object_or_404(CompanyProfile, pk=pk)
        uploaded_file = request.FILES.get('pdf')

        if uploaded_file:
            # Dateiinhalt lesen
            content = uploaded_file.read()

            # SHA-256 Hash berechnen
            sha256 = hashlib.sha256(content).hexdigest()

            # Nur manuelle Uploads prüfen
            pdf = PDFFile.objects.filter(file_hash=sha256, source='manual').first()

            if pdf:
                PDFScrapeDate.objects.create(pdf_file=pdf)
                print(f"Duplicate gefunden, Metadaten aktualisiert: {uploaded_file.name}")
            else:
                new_pdf = PDFFile(company=company, source='manual')
                new_pdf.file.save(uploaded_file.name, ContentFile(content))
                new_pdf.save()
                PDFScrapeDate.objects.create(pdf_file=new_pdf)
                print(f"Neues PDF gespeichert: {uploaded_file.name}")

        return redirect('company_detail', pk=pk)


class CompanyCreateUpdateView(View):
    template_name = 'api/company_form.html'

    def get(self, request, pk=None):
        instance = get_object_or_404(CompanyProfile, pk=pk) if pk else None
        form = CompanyProfileForm(instance=instance)
        formset = CompanyURLFormSet(instance=instance)
        form_action = reverse('company_edit', args=[instance.pk]) if instance else reverse('company_create')
        return render(request, self.template_name, {
            'form': form,
            'formset': formset,
            'instance': instance,
            'form_action': form_action
        })

    def post(self, request, pk=None):
        instance = get_object_or_404(CompanyProfile, pk=pk) if pk else None
        form = CompanyProfileForm(request.POST, instance=instance)
        formset = CompanyURLFormSet(request.POST, instance=instance)

        if form.is_valid() and formset.is_valid():
            company = form.save()
            formset.instance = company
            formset.save()

            # 🔁 Wenn AJAX-Request → JSON statt Redirect
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True})
            
            return redirect('dashboard')  # klassisch für Nicht-AJAX (z. B. aus dem Dashboard)

        # Fehlerfall: AJAX → HTML-Response in JSON
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            html = render_to_string(self.template_name, {
                'form': form,
                'formset': formset,
                'instance': instance
            }, request=request)
            return JsonResponse({'success': False, 'html': html})

        # Nicht-AJAX: klassisch rendern
        return render(request, self.template_name, {
            'form': form,
            'formset': formset,
            'instance': instance
        })


class QueryCreateUpdateView(View):
    def get(self, request, pk=None):
        if pk:
            query = get_object_or_404(Query, pk=pk)
            form = QueryForm(instance=query)
        else:
            query = None
            form = QueryForm()
        return render(request, 'api/query_form.html', {
            'form': form,
            'instance': query,
            'form_action': request.path
        })

    def post(self, request, pk=None):
        if pk:
            query = get_object_or_404(Query, pk=pk)
            form = QueryForm(request.POST, instance=query)
        else:
            form = QueryForm(request.POST)

        if form.is_valid():
            form.save()
            queries = Query.objects.all()
            html = render_to_string('api/query_table.html', {'queries': queries})
            return JsonResponse({'success': True, 'html': html})
        else:
            html = render_to_string('api/query_form.html', {
                'form': form,
                'form_action': request.path
            })
            return JsonResponse({'success': False, 'html': html})


class ScrapeTriggerView(View):
    def get(self, request):
        """Check scrape status."""
        return JsonResponse({"scraping": bool(cache.get(SCRAPING_STATUS_KEY))})

    def post(self, request):
        if cache.get(SCRAPING_STATUS_KEY):
            return JsonResponse({"status": "already running"}, status=400)

        cache.set(SCRAPING_STATUS_KEY, True, timeout=3600)

        def run_scraping():
            try:
                active_companies = CompanyProfile.objects.filter(active=True).prefetch_related('urls')

                for company in active_companies:
                    urls = company.urls.filter(active=True)
                    for url in urls:
                        try:
                            print(f"Starte Scraper für Firma {company.name} mit URL: {url.url}")
                            call_command('run_scraper', domain=url.url, company_id=company.id)
                        except Exception as cmd_error:
                            print(f"Fehler beim Scrapen von {company.name} / {url.url}: {cmd_error}")
                            traceback.print_exc()

                # Scrape-Zeitstempel aktualisieren
                CompanyProfile.objects.filter(active=True).update(last_scraped=now())
                cache.delete(SCRAPING_STATUS_KEY)

                print("Scraping abgeschlossen.")
            except Exception as e:
                print("Fehler im Scrape-Thread:")
                traceback.print_exc()
                cache.delete(SCRAPING_STATUS_KEY)

        threading.Thread(target=run_scraping).start()

        companies = CompanyProfile.objects.filter(active=True).values('id', 'last_scraped')
        return JsonResponse({
            "status": "started",
            "companies": list(companies)
        })



class TriggerEvaluationView(View):
    def get(self, request):
        running = bool(cache.get(EVALUATION_STATUS_KEY))
        return JsonResponse({"running": running})

    def post(self, request):
        if cache.get(EVALUATION_STATUS_KEY):
            return JsonResponse({"status": "already running"}, status=400)

        cache.set(EVALUATION_STATUS_KEY, True, timeout=3600)

        def run_evaluation():
            try:
                import time
                time.sleep(15)  # Simuliere Auswertung

                # Beispiel: genutzte Query-IDs
                used_query_ids = [1, 3, 7]
                queries = Query.objects.filter(id__in=used_query_ids)

                # Für jede Firma und jede Query ein leeres EvaluationResult anlegen
                for company in CompanyProfile.objects.filter(active=True):
                    for query in queries:
                        EvaluationResult.objects.create(
                            query=query,
                            company=company,
                            pdf_file=None,  # falls noch nicht vorhanden
                            result_data={},  # leeres Ergebnis
                            timestamp=now()
                        )

                cache.delete(EVALUATION_STATUS_KEY)
            except Exception:
                import traceback
                traceback.print_exc()
                cache.delete(EVALUATION_STATUS_KEY)

        threading.Thread(target=run_evaluation).start()

        # Rückgabe der letzten Evaluationszeitpunkte pro Firma (aggregiert)
        # Beispiel: Hole das neueste Ergebnis je Firma
        from django.db.models import Max
        latest = (EvaluationResult.objects
            .values('company_id')
            .annotate(last_evaluated=Max('timestamp'))
            .filter(company_id__in=CompanyProfile.objects.filter(active=True).values_list('id', flat=True))
        )

        companies = []
        for item in latest:
            companies.append({
                'id': item['company_id'],
                'last_evaluated': item['last_evaluated'],
            })

        # Am Ende deiner Ausführung
        response_data = [
            {
                "company_id": item["id"],
                "last_evaluated": item["last_evaluated"].strftime("%d.%m.%Y %H:%M") if item["last_evaluated"] else None
            }
            for item in companies
        ]

        return JsonResponse({
            "status": "started",
            "updated_companies": response_data
        })

class LLMRunEvaluationView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Evaluator with real dependencies
        self.db_interface = LLMDBInterface()
        self.llm_provider = HuggingFaceLLMProvider()  # Replace with your desired LLM
        self.evaluator = LLMEvaluator(db_interface=self.db_interface, llm_provider=self.llm_provider)
    def post(self, request):
        serializer = LLMQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        # Run evaluation
        result = self.evaluator.evaluate(
            pdf_path=data['pdf_path'],
            query=data['query'], #TODO: the question itself
            query_id=data['query_id'],
            company_id=data['company_id'],
            user=data.get('user')
        )
        result_serializer = LLMResultSerializer(result)
        return Response(result_serializer.data)









@require_POST
def toggle_pdf_active(request):
    pdf_id = request.POST.get('pdf_id')
    use_for_analysis = request.POST.get('active') == 'true'

    try:
        pdf = PDFFile.objects.get(id=pdf_id)
        pdf.active = use_for_analysis
        pdf.save()
        return JsonResponse({'success': True})
    except PDFFile.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'PDF nicht gefunden'}, status=404)

@require_POST
@csrf_exempt
def toggle_query_active(request, pk):
    try:
        data = json.loads(request.body.decode('utf-8'))
        active = data.get('active')
        query = get_object_or_404(Query, pk=pk)
        query.active = active
        query.save()
        return JsonResponse({'success': True, 'active': query.active})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
def toggle_company_active(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            company_id = data.get('company_id')
            active = data.get('active')

            company = CompanyProfile.objects.get(id=company_id)
            company.active = active
            company.save()

            return JsonResponse({'status': 'success'})
        except CompanyProfile.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Company not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
