#!/bin/sh
set -eu

# DOMAINS=comma-separated (e.g. trade.example.com,www.trade.example.com) or single DOMAIN (legacy).
LIST="${DOMAINS:-}"
if [ -z "$LIST" ]; then
  LIST="${DOMAIN:-}"
fi
if [ -z "$LIST" ]; then
  echo "error: set DOMAINS (comma-separated hostnames) or DOMAIN in .env" >&2
  exit 1
fi

# First hostname: default Let's Encrypt live dir name when using certbot -d first -d second ...
PRIMARY_DOMAIN=$(printf '%s' "$LIST" | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
# Nginx server_name: space-separated
SERVER_NAMES=$(printf '%s' "$LIST" | tr ',' ' ' | tr -s ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

export PRIMARY_DOMAIN SERVER_NAMES

if [ -f "/etc/letsencrypt/live/${PRIMARY_DOMAIN}/fullchain.pem" ]; then
  echo "nginx: TLS cert found at live/${PRIMARY_DOMAIN}/, using HTTPS config (server_name: ${SERVER_NAMES})"
  envsubst '$PRIMARY_DOMAIN $SERVER_NAMES' </templates/default-ssl.conf.template >/etc/nginx/conf.d/default.conf
else
  echo "nginx: no TLS cert yet for live/${PRIMARY_DOMAIN}/, using HTTP-only config (server_name: ${SERVER_NAMES})"
  envsubst '$PRIMARY_DOMAIN $SERVER_NAMES' </templates/default-init.conf.template >/etc/nginx/conf.d/default.conf
fi

exec nginx -g 'daemon off;'
