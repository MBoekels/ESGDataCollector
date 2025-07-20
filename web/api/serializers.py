"""
Serializers for LLM evaluation API and data models.
Makes the Django models available as REST API endpoints.
"""

from rest_framework import serializers
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from .models import (
    CompanyProfile,
    CompanyURL,
    EvaluationResult,
    PDFFile,
    PDFOriginURL,
    PDFScrapeDate,
    Query,
)

# Serializers for specific API endpoints (e.g., LLM evaluation)
class LLMQuerySerializer(serializers.Serializer):
    query = serializers.CharField()
    query_id = serializers.IntegerField()
    company_id = serializers.IntegerField()
    user = serializers.CharField(required=False, allow_null=True)

class LLMDataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationResult
        fields = [
            'id', 'company', 'query', 'pdf_file', 'timestamp',
            'answer', 'chunk_id', 'chunk_type',
            'report_year', 'cosine_similarity', 'confidence',
            'references', 'model_version'
        ]


# --- Model Serializers ---
# These serializers are used to expose the data models via the REST API.

class CompanyURLSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompanyURL
        fields = "__all__"


class PDFScrapeDateSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFScrapeDate
        fields = "__all__"


class PDFOriginURLSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFOriginURL
        fields = "__all__"


class QuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = Query
        fields = "__all__"


class EvaluationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationResult
        fields = "__all__"


class PDFFileSerializer(serializers.ModelSerializer):
    scrape_dates = PDFScrapeDateSerializer(many=True, read_only=True)
    origin_urls = PDFOriginURLSerializer(many=True, read_only=True)
    
    class Meta:
        model = PDFFile
        fields = (
            "id", "company", "file", "file_hash", "file_size", "source", "active",
            "chunk_vector_index_path", "processing_status", "scrape_dates",
            "origin_urls", "vector",
        )
        # These fields are calculated automatically in the model's save method.
        read_only_fields = ("file_hash", "file_size")


class CompanyProfileSerializer(serializers.ModelSerializer):
    urls = CompanyURLSerializer(many=True, read_only=True)
    pdfs = PDFFileSerializer(many=True, read_only=True)

    class Meta:
        model = CompanyProfile
        fields = (
            "id", "name", "industry", "info", "active", "last_scraped", "urls", "pdfs"
        )


# Utility to create a token for a user (run once per user)
def create_token_for_user(username):
    user = User.objects.get(username=username)
    token, created = Token.objects.get_or_create(user=user)
    return token.key
