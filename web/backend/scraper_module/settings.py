BOT_NAME = 'scraper_module'

SPIDER_MODULES    = ['backend.scraper_module.spiders']  
NEWSPIDER_MODULE = 'backend.scraper_module.spiders'

DOWNLOADER_MIDDLEWARES = {
    'backend.scraper_module.middlewares.DynamicDelayRetryMiddleware': 543,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
}

RETRY_ENABLED = True
RETRY_TIMES = 5
