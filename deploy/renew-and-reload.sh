#!/bin/sh
# Renew Let’s Encrypt certs (webroot) and reload Nginx. Run from host cron after deploy.
set -eu
cd "$(dirname "$0")/.."
docker compose -f docker-compose.prod.yml --env-file .env run --rm certbot renew --webroot -w /var/www/certbot
docker compose -f docker-compose.prod.yml --env-file .env exec nginx nginx -s reload
