import asyncio
import time
import random
import logging
from typing import Callable, Any, TypeVar

from config import GEMINI_RPM_LIMIT

logger = logging.getLogger("rate_limiter")

T = TypeVar("T")

class GracefulQuotaExceededError(Exception):
    """Raised when the daily quota of the Gemini API is exhausted."""
    pass

class AsyncRateLimiter:
    """
    An async token-bucket rate limiter to throttle requests to the Gemini API.
    Ensures that we do not exceed the configured Requests Per Minute (RPM) limit.
    """
    def __init__(self, requests_per_minute: int = GEMINI_RPM_LIMIT):
        self.rate = requests_per_minute
        self.interval = 60.0
        self.tokens = float(requests_per_minute)
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire a token from the bucket. Blocks if no tokens are available.
        """
        # If rate is set to 0 or negative, disable rate limiting
        if self.rate <= 0:
            return

        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_refill
                # Refill tokens based on elapsed time
                refill_amount = elapsed * (self.rate / self.interval)
                if refill_amount > 0:
                    self.tokens = min(float(self.rate), self.tokens + refill_amount)
                    self.last_refill = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                
                # Calculate sleep duration to wait for the next token
                sleep_time = (1.0 - self.tokens) * (self.interval / self.rate)
                await asyncio.sleep(sleep_time)

# Global rate limiter instance
global_rate_limiter = AsyncRateLimiter()

async def execute_with_retry(
    func: Callable[..., Any], 
    *args, 
    max_retries: int = 5, 
    initial_backoff: float = 1.0, 
    **kwargs
) -> Any:
    """
    Executes an async function, applying exponential backoff with jitter if 
    a rate limit (429) or quota limit is encountered.
    Detects daily quota limits and exits gracefully.
    """
    # Always acquire a rate limiter token first to protect RPM limits
    await global_rate_limiter.acquire()
    
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            # Execute target function (could be sync or async, but we treat it as async/blocking-async)
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            
            # 1. Detect Daily Quota Exhaustion (typically "daily" or "day" or "per_day" or "requestsperday")
            is_daily_quota = (
                "quota exceeded" in err_lower and 
                ("daily" in err_lower or "per_day" in err_lower or "requestsperday" in err_lower or ("day" in err_lower and "minute" not in err_lower))
            )
            
            if is_daily_quota:
                logger.error("❌ Daily Gemini quota exceeded. Stopping operations gracefully.")
                raise GracefulQuotaExceededError("Daily Gemini API quota limit exceeded.") from e

            # 2. Detect 429 Rate Limits / Resource Exhaustion or 503 Service Unavailable
            is_retryable = (
                "429" in err_str or 
                "resource_exhausted" in err_lower or 
                "rate limit" in err_lower or
                "503" in err_str or
                "unavailable" in err_lower or
                "service unavailable" in err_lower
            )
            
            if is_retryable:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Rate limit retry budget exhausted after {max_retries} attempts.")
                    raise e
                
                # Apply backoff with jitter
                jitter = random.uniform(0.8, 1.2)
                sleep_duration = backoff * jitter
                logger.warning(f"⚠️ [WARNING] Transient API error (429/503) hit. Retrying in {sleep_duration:.2f}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(sleep_duration)
                backoff *= 2.0
            else:
                # Other exceptions are re-raised immediately without retrying
                raise e
