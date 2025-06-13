import scrapy
import os
import pandas as pd
from urllib.parse import urlparse, urljoin

class PdfSpider(scrapy.Spider):
    name = "pdf_spider"

    def __init__(self, domain, save_folder='pdfs', *args, **kwargs):
        super(PdfSpider, self).__init__(*args, **kwargs)
        self.domain = domain
        self.allowed_domains = [urlparse(domain).netloc]
        self.start_urls = [domain]
        self.save_folder = save_folder
        self.pdf_files = []
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def parse(self, response):
        content_type = response.headers.get('Content-Type', '').decode('utf-8')

        if 'application/pdf' in content_type:
            yield from self.save_pdf(response)
        elif 'text/html' in content_type:
            for link in response.css('a::attr(href)').getall():
                full_url = urljoin(response.url, link)
                if self.is_internal_link(full_url):
                    if full_url.endswith('.pdf'):
                        yield response.follow(full_url, self.save_pdf)
                    else:
                        yield response.follow(full_url, self.parse)


    def save_pdf(self, response):
        pdf_url = response.url
        pdf_name = pdf_url.split('/')[-1]

        upload_date = response.headers.get('Last-Modified', '').decode('utf-8')

        self.pdf_files.append({
            'file_name': pdf_name,
            'file_path': pdf_url,
            'upload_date': upload_date
        })

        self.log(f'Scraped: {pdf_name}')
        yield {}  # Dummy-Yield, damit es ein Generator ist


    def is_internal_link(self, url):
        return urlparse(url).netloc in self.allowed_domains
    
    def closed(self, reason):
        df = pd.DataFrame(self.pdf_files)
        csv_path = os.path.join(self.save_folder, 'pdf_files.csv')
        df.to_csv(csv_path, index=False)
        self.log(f"Finished scraping PDFs. File paths saved to {csv_path}")