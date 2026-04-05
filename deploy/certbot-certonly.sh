#!/bin/sh
# First-time Let's Encrypt certificate using hostnames from .env (DOMAINS or DOMAIN) and CERTBOT_EMAIL.
# Run from the repo root after: docker compose -f docker-compose.prod.yml --env-file .env up -d
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
COMPOSE_FILE="$ROOT/docker-compose.prod.yml"

if [ ! -f "$ENV_FILE" ]; then
  echo "error: missing $ENV_FILE" >&2
  exit 1
fi

# Read KEY=value without sourcing the whole file (avoids shell metacharacters in unrelated keys).
get_val() {
  _key="$1"
  grep -E "^[[:space:]]*${_key}[[:space:]]*=" "$ENV_FILE" 2>/dev/null | head -1 | sed "s/^[[:space:]]*${_key}[[:space:]]*=//" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//;s/^'"'"'//;s/'"'"'$//'
}

DOMAINS_CSV=$(get_val DOMAINS)
if [ -z "$DOMAINS_CSV" ]; then
  DOMAINS_CSV=$(get_val DOMAIN)
fi
if [ -z "$DOMAINS_CSV" ]; then
  echo "error: set DOMAINS (comma-separated) or DOMAIN in $ENV_FILE" >&2
  exit 1
fi

CERTBOT_EMAIL_VAL=$(get_val CERTBOT_EMAIL)
if [ -z "$CERTBOT_EMAIL_VAL" ]; then
  echo "error: set CERTBOT_EMAIL in $ENV_FILE" >&2
  exit 1
fi

CERTBOT_D_ARGS=""
old_ifs=$IFS
IFS=,
for d in $DOMAINS_CSV; do
  d=$(printf '%s' "$d" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [ -z "$d" ] && continue
  CERTBOT_D_ARGS="$CERTBOT_D_ARGS -d $d"
done
IFS=$old_ifs

if [ -z "$CERTBOT_D_ARGS" ]; then
  echo "error: no hostnames parsed from DOMAINS/DOMAIN" >&2
  exit 1
fi

echo "Running certbot certonly with:$CERTBOT_D_ARGS"
cd "$ROOT"
# shellcheck disable=SC2086
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  $CERTBOT_D_ARGS \
  --email "$CERTBOT_EMAIL_VAL" \
  --agree-tos \
  --non-interactive
