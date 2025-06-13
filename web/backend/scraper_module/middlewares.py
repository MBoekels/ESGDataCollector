import time
import random
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message

class DynamicDelayRetryMiddleware(RetryMiddleware):
    def __init__(self, settings):
        super().__init__(settings)
        self.base_delay = 1  # Startverzögerung in Sekunden
        self.max_delay = 60  # Maximaler Delay
        self.retry_http_codes = set([403, 429])  # Fehlercodes, die verzögert werden

    def process_response(self, request, response, spider):
        if response.status in self.retry_http_codes:
            retries = request.meta.get('retry_times', 0) + 1
            delay = min(self.base_delay * (2 ** (retries - 1)), self.max_delay)
            jitter = random.uniform(0, 1.5)
            total_delay = delay + jitter

            spider.logger.warning(
                f"Blocked with {response.status}. Retrying {request.url} in {total_delay:.2f}s (attempt {retries})"
            )

            time.sleep(total_delay)

            reason = response_status_message(response.status)
            return self._retry(request, reason, spider) or response

        return response
