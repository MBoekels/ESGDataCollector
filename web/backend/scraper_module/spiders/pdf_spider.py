import scrapy
import os
import pandas as pd
from urllib.parse import urlparse, urljoin

class PdfSpider(scrapy.Spider):
    name = "pdf_spider"

    def __init__(self, domain, save_folder='pdfs', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain
        self.allowed_domains = [urlparse(domain).netloc]
        self.start_urls = [domain]
        self.save_folder = save_folder
        self.pdf_files = []
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Hier definierst du deine Browser-Header
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/114.0.0.0 Safari/537.36',
            'Referer': self.domain
        }

    def parse(self, response):
        ct = response.headers.get('Content-Type', b'').decode('utf-8')
        if 'application/pdf' in ct:
            yield from self.save_pdf(response)
        elif 'text/html' in ct:
            for link in response.css('a::attr(href)').getall():
                full_url = urljoin(response.url, link)
                if self.is_internal_link(full_url):
                    if full_url.endswith('.pdf'):
                        # Hier: Folge dem Link mit deinen Headern
                        yield scrapy.Request(
                            url=full_url,
                            callback=self.save_pdf,
                            headers=self.headers
                        )
                    else:
                        yield response.follow(full_url, self.parse)

    def save_pdf(self, response):
        pdf_name = response.url.split('/')[-1]
        upload_date = response.headers.get('Last-Modified', b'').decode('utf-8')

        # Speichere lokale Datei direkt
        local_path = os.path.join(self.save_folder, pdf_name)
        with open(local_path, 'wb') as f:
            f.write(response.body)

        self.pdf_files.append({
            'file_name': pdf_name,
            'file_path': local_path,   # jetzt lokaler Pfad
            'upload_date': upload_date
        })

        self.log(f'Saved locally: {pdf_name}')
        yield {}  # Dummy

    def is_internal_link(self, url):
        return urlparse(url).netloc in self.allowed_domains

    def closed(self, reason):
        df = pd.DataFrame(self.pdf_files)
        csv_path = os.path.join(self.save_folder, 'pdf_files.csv')
        df.to_csv(csv_path, index=False)
        self.log(f"Finished scraping PDFs. File paths saved to {csv_path}")
