#!/bin/sh
# With explicit certbot subcommand: run once (e.g. `docker compose run --rm certbot certonly ...`).
# With no args, or only image default flags like --help: renewal loop (`docker compose up`).
set -eu

if [ "$#" -gt 0 ]; then
  case "$1" in
  -*)
    ;;
  certonly|certificates|revoke|register|plugins|delete|update_account|install|rollback|run|show_account|renew|update_symlinks|enhance)
    exec certbot "$@"
    ;;
  esac
fi

trap exit TERM
while :; do
  certbot renew --webroot -w /var/www/certbot || true
  sleep 12h &
  wait "$!"
done
