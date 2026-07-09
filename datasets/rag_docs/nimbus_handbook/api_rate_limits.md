# API Rate Limits

The Nimbus API allows 1,000 requests per hour per access token on Free and Personal plans,
and 10,000 requests per hour on Business and Enterprise plans. Rate limit status is returned
in the `X-RateLimit-Remaining` response header on every request.

Exceeding the rate limit returns an HTTP 429 response with a `Retry-After` header indicating
how many seconds to wait. Sustained rate-limit violations (more than 10 consecutive 429
responses) may result in temporary token suspension; contact support to have a suspended
token reinstated.
