import os
import hashlib
import pandas as pd
import requests
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from backend.scraper_module.spiders.pdf_spider import PdfSpider
from api.models import CompanyProfile, PDFFile, PDFScrapeDate, PDFOriginURL

class Command(BaseCommand):
    help = 'Run the PDF scraper for a company and save results to the database'

    def add_arguments(self, parser):
        parser.add_argument('--domain', required=True, type=str, help='The domain to crawl for PDFs')
        parser.add_argument('--company_id', required=True, type=int, help='The ID of the company to associate with the PDFs')
        parser.add_argument('--save_folder', default='media/pdfs', type=str, help='Local folder to save CSV & PDFs')

    def handle(self, *args, **options):
        domain = options['domain']
        company_id = options['company_id']
        output_folder = options['save_folder']

        # 1) Company existenz prüfen
        try:
            company = CompanyProfile.objects.get(id=company_id)
        except CompanyProfile.DoesNotExist:
            self.stderr.write(f"Company with ID {company_id} does not exist.")
            return

        # 2) Ausgabe-Ordner anlegen
        os.makedirs(output_folder, exist_ok=True)

        # 3) Scrapy-Settings laden (aus backend.scraper_module.settings)
        os.environ.setdefault('SCRAPY_SETTINGS_MODULE', 'backend.scraper_module.settings')
        settings = get_project_settings()

        # Optional: Overrides an Settings, z.B. Verzeichnis für Dateien
        settings.set('FILES_STORE', output_folder, priority='cmdline')

        # 4) CrawlerProcess mit diesen Settings starten
        process = CrawlerProcess(settings)
        process.crawl(PdfSpider, domain=domain, save_folder=output_folder)
        self.stdout.write(f"Starte In-Process Scrapy für {domain}…")
        process.start()  # blockiert, bis Spider fertig

        # 5) Nach dem Crawl: CSV einlesen und in DB speichern
        csv_path = os.path.join(output_folder, 'pdf_files.csv')
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            local_path = row['file_path']          # statt pdf_url
            file_name  = row['file_name']

            try:
                # PDF direkt von Platte laden:
                with open(local_path, 'rb') as f:
                    content = f.read()

                sha256 = hashlib.sha256(content).hexdigest()
                pdf = PDFFile.objects.filter(file_hash=sha256, source='webscraped').first()

                if pdf:
                    PDFScrapeDate.objects.create(pdf_file=pdf)
                    if not pdf.origin_urls.filter(url=local_path).exists():
                        PDFOriginURL.objects.create(pdf_file=pdf, url=local_path)
                    self.stdout.write(f"Duplicate gefunden, Metadaten aktualisiert: {file_name}")
                else:
                    new_pdf = PDFFile(company=company, source='webscraped')
                    new_pdf.file.save(file_name, ContentFile(content))
                    new_pdf.save()
                    PDFScrapeDate.objects.create(pdf_file=new_pdf)
                    PDFOriginURL.objects.create(pdf_file=new_pdf, url=local_path) # TODO Local path auf url anpassen, pdf_files.csv muss angepasst werden
                    self.stdout.write(f"Neues PDF gespeichert: {file_name}")

            except Exception as e:
                self.stderr.write(f"Fehler beim Lesen {local_path}: {e}")

        self.stdout.write("Scraping & Speichern abgeschlossen.")
