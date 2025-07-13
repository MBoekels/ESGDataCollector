"""
Serializers for LLM evaluation API.
"""

from rest_framework import serializers
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User

class LLMQuerySerializer(serializers.Serializer):
    pdf_path = serializers.CharField()
    query = serializers.CharField()
    query_id = serializers.IntegerField()
    company_id = serializers.IntegerField()
    user = serializers.CharField(required=False, allow_null=True)

class LLMResultSerializer(serializers.Serializer):
    query_id = serializers.IntegerField()
    company_id = serializers.IntegerField()
    timestamp = serializers.CharField()
    references = serializers.ListField()
    per_query_data = serializers.DictField()
    user = serializers.CharField(allow_null=True)
    model_version = serializers.CharField(allow_null=True)
    processing_time = serializers.FloatField(allow_null=True)

# Utility to create a token for a user (run once per user)
def create_token_for_user(username):
    user = User.objects.get(username=username)
    token, created = Token.objects.get_or_create(user=user)
    return token.key
