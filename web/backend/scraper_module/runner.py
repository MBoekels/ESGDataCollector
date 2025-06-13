import os
import django
import requests
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from django.core.files.base import ContentFile
from ESGDataCollector.web.backend.scraper_module.spiders.pdf_spider import PdfSpider
from api.models import CompanyProfile, PDFFile

from crochet import setup, wait_for

# Django Setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

setup()

def save_pdf_to_db(file_data, company: CompanyProfile):
    """
    Speichert ein PDF aus dem Webscraping direkt in die Datenbank inklusive Dateiinhalt.
    """
    # PDF herunterladen
    response = requests.get(file_data['file_path'])
    if response.status_code != 200:
        print(f"❌ Fehler beim Download: {file_data['file_path']}")
        return

    file_content = ContentFile(response.content)
    filename = file_data['file_name']

    # Neues PDFFile-Objekt erstellen
    pdf = PDFFile(
        company=company,
        source='webscraped',
        file_size=len(response.content),
        extraction_success=False,  # wird ggf. später durch LLM geändert
    )
    pdf.file.save(filename, file_content, save=True)
    print(f"✅ Gespeichert: {filename}")


class DjangoAwarePdfSpider(PdfSpider):
    """
    Subclass of your PdfSpider with Django integration for saving to DB.
    """
    def __init__(self, domain, company_id, *args, **kwargs):
        super().__init__(domain, *args, **kwargs)
        self.company = CompanyProfile.objects.get(id=company_id)

    def closed(self, reason):
        # override to save to DB instead of CSV
        for file_data in self.pdf_files:
            save_pdf_to_db(file_data, self.company)
        print(f"✅ Finished scraping for company: {self.company.name}")


@wait_for(timeout=60.0)
def run(domain: str, company_id: int):
    process = CrawlerProcess(get_project_settings())
    process.crawl(DjangoAwarePdfSpider, domain=domain, company_id=company_id)
    process.start()


# --- Optional: Kommandozeilen-Start (z. B. `python run_scraper.py https://example.com 1`)
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python run_scraper.py <domain> <company_id>")
    else:
        domain = sys.argv[1]
        company_id = int(sys.argv[2])
        run(domain, company_id)
