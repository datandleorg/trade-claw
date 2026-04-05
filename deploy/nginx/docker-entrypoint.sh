#!/bin/sh
set -eu

if [ -z "${DOMAIN:-}" ]; then
  echo "error: DOMAIN is not set" >&2
  exit 1
fi

if [ -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
  echo "nginx: TLS cert found for ${DOMAIN}, using HTTPS config"
  envsubst '$DOMAIN' </templates/default-ssl.conf.template >/etc/nginx/conf.d/default.conf
else
  echo "nginx: no TLS cert yet for ${DOMAIN}, using HTTP-only config (run certbot certonly, then restart nginx)"
  envsubst '$DOMAIN' </templates/default-init.conf.template >/etc/nginx/conf.d/default.conf
fi

exec nginx -g 'daemon off;'
