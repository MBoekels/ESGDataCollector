from django.contrib import admin
from .models import PDFFile

@admin.register(PDFFile)
class PDFFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'company', 'source', 'active')
    readonly_fields = ('id',)
