"""
File: web/api/models.py

Role:
    This file defines the database schema for the application using the Django Object-Relational Mapper (ORM).
    It includes models for `CompanyProfile`, `PDFFile`, `Query`, `EvaluationResult`, and `PDFVector`, which
    are the core data structures for storing company information, uploaded documents, user queries, LLM
    evaluation outputs, and document-level embeddings.

Interactions:
    - `web/api/views.py`: The views use these models to handle CRUD (Create, Read, Update, Delete) operations
      triggered by API requests from the frontend.
    - `web/backend/llm_module/llm_db_interface.py`: Uses `PDFFile` and `PDFVector` to manage PDF embeddings.
    - `web/backend/llm_module/evaluator.py`: The results from this module are stored in the `EvaluationResult` model.
"""
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import URLValidator
from django.utils.timezone import now
from django.contrib.postgres.fields import JSONField  # for Postgres, else use models.JSONField in Django 3.1+
import hashlib


class CompanyProfile(models.Model):
    name = models.CharField(max_length=255, unique=True)
    industry = models.CharField(max_length=255, blank=True)
    info = models.TextField(blank=True)
    active = models.BooleanField(default=True)
    last_scraped = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return self.name

class CompanyURL(models.Model):
    company = models.ForeignKey(CompanyProfile, related_name='urls', on_delete=models.CASCADE)
    url = models.URLField(validators=[URLValidator()],  max_length=2000)
    added_on = models.DateTimeField(auto_now_add=True)
    is_valid = models.BooleanField(default=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.url


class PDFFile(models.Model):
    company = models.ForeignKey("CompanyProfile", related_name='pdfs', on_delete=models.CASCADE)
    file = models.FileField(upload_to='pdfs/')
    file_hash = models.CharField(max_length=64, unique=True)  # SHA-256 hash
    file_size = models.IntegerField(null=True, blank=True)
    source = models.CharField(max_length=20, choices=[('manual', 'Manual'), ('webscraped', 'Webscraped')])
    report_year = models.IntegerField(null=True, blank=True)
    active = models.BooleanField(default=True)
    chunk_vector_index_path = models.CharField(max_length=512, blank=True, null=True)
    document_vector_index_path = models.CharField(max_length=255, blank=True, null=True)
    processing_status = models.CharField(max_length=20, choices=[('pending', 'Pending'), ('success', 'Success'), ('failed', 'Failed')], default='pending')

    def __str__(self):
        return f"{self.file.name} ({self.company.name})"

    def save(self, *args, **kwargs):
        if not self.file_hash and self.file:
            self.file.seek(0)
            sha256 = hashlib.sha256()
            for chunk in self.file.chunks():
                sha256.update(chunk)
            self.file_hash = sha256.hexdigest()
        if not self.file_size and self.file:
            self.file_size = self.file.size
        super().save(*args, **kwargs)
    
    def latest_scrape_date(self):
        return self.scrape_dates.order_by('-scraped_at').first()

    def latest_origin_url(self):
        return self.origin_urls.order_by('-id').first()



class PDFScrapeDate(models.Model):
    pdf_file = models.ForeignKey(PDFFile, related_name='scrape_dates', on_delete=models.CASCADE)
    scraped_at = models.DateTimeField(default=now)


class PDFOriginURL(models.Model):
    pdf_file = models.ForeignKey(PDFFile, related_name='origin_urls', on_delete=models.CASCADE)
    url = models.URLField()

    class Meta:
        unique_together = ('pdf_file', 'url')


class Query(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    question = models.TextField()
    enriched_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_edited = models.DateTimeField(auto_now=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

class EvaluationResult(models.Model):
    """Stores a single RAG-based result for a query and a chunk from a PDF."""
    query = models.ForeignKey("Query", on_delete=models.CASCADE, related_name='results')
    company = models.ForeignKey("CompanyProfile", on_delete=models.CASCADE, related_name='results')
    pdf_file = models.ForeignKey("PDFFile", on_delete=models.CASCADE, related_name='results', help_text="The source PDF for this result.")

    timestamp = models.DateTimeField(default=now)

    # RAG fields
    chunk_id = models.CharField(max_length=255)
    chunk_type = models.CharField(max_length=50)
    answer = models.TextField()
    report_year = models.IntegerField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    cosine_similarity = models.FloatField(null=True, blank=True)

    # Reference metadata
    references = models.JSONField(help_text="Metadata from chunk used to produce this result.", default=dict)

    # Meta / LLM info
    model_version = models.CharField(max_length=255, blank=True, null=True)
    processing_time = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"Result for query {self.query_id} on PDF {self.pdf_file_id} (chunk {self.chunk_id})"

