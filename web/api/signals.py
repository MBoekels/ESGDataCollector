import traceback

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import PDFFile
from backend.llm_module.pdf_preprocessor import PDFPreprocessor
from backend.llm_module.pdf_preprocessor import process_and_embed_pdf_task

@receiver(post_save, sender=PDFFile)
def pdf_file_post_save(sender, instance, created, **kwargs):
    """
    Trigger PDF pre-processing and embedding after a new PDFFile is created.
    """
    if created and instance.processing_status == 'pending':
        print(f"PDFFile created (ID: {instance.id}), queuing for pre-processing.")
        process_and_embed_pdf_task.delay(instance.id)