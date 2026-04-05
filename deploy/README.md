# Production deploy (Docker Compose + Nginx + Certbot)

This stack uses [docker-compose.prod.yml](../docker-compose.prod.yml): **Redis**, **Streamlit**, **Celery worker**, **Celery beat**, **Nginx** (ports 80/443), and a **Certbot** sidecar that runs `certbot renew` on a loop. Only Nginx is exposed on the host; Redis and Streamlit stay on the internal Docker network.

## Prerequisites

- A DigitalOcean droplet (or similar) with **Docker** and the **Docker Compose plugin** installed.
- A **domain** whose **A record** points at the droplet’s public IPv4 (and optionally **AAAA** for IPv6 if you enable it end-to-end).
- **Kite redirect URL** in the [Kite developer dashboard](https://kite.trade/dashboard) set to `https://YOUR_DOMAIN/` (and `/callback` if you use that path).

## 1. DNS and firewall

1. Create an **A** record: `YOUR_DOMAIN` → droplet IP (add `www` if you want that hostname too).
2. On the droplet, allow SSH and HTTP/S, then enable the firewall:

   ```bash
   sudo ufw allow OpenSSH
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

## 2. Environment

In the project root, copy [.env.example](../.env.example) to `.env` and set at least:

- `KITE_API_KEY`, `KITE_API_SECRET`, and any other app keys you use (see `.env.example`).
- **`DOMAINS`** — comma-separated hostnames Nginx will answer on (no spaces, or spaces trimmed), e.g. `trade.example.com,www.trade.example.com`. **Alternatively**, set a single **`DOMAIN`** (legacy) if you only need one hostname.
- **`CERTBOT_EMAIL`** — used only for the **first** `certbot certonly` (Let’s Encrypt account / expiry notices).

Example (apex + `www`):

```bash
DOMAINS=trade.example.com,www.trade.example.com
CERTBOT_EMAIL=you@example.com
```

**Let’s Encrypt certificate path:** the files under `/etc/letsencrypt/live/<name>/` use **the first** hostname in `DOMAINS` as `<name>` by default when you pass `-d` in that same order to `certbot certonly`. Keep the first entry in `DOMAINS` identical to the first `-d` you used when issuing the cert.

## 3. Build and start

From the repository root:

```bash
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
```

On first boot, Nginx uses an **HTTP-only** config (ACME webroot + proxy to Streamlit) because certificates are not present yet. You can open `http://YOUR_DOMAIN` to confirm the app loads.

## 4. Obtain the first TLS certificate

Run Certbot **once** against the shared webroot (Nginx must be up).

**Recommended:** hostnames and email are taken from `.env` (`DOMAINS` or `DOMAIN`, plus `CERTBOT_EMAIL`):

```bash
./deploy/certbot-certonly.sh
```

The script turns `DOMAINS=trade.example.com,www.trade.example.com` into multiple `-d` flags. The **first** hostname is the default Let’s Encrypt `live/<name>/` directory (keep the same order as in `.env`).

**Manual** (same effect, if you prefer not to use the script):

```bash
docker compose -f docker-compose.prod.yml --env-file .env run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  -d trade.example.com \
  -d www.trade.example.com \
  --email you@example.com \
  --agree-tos \
  --non-interactive
```

Then **restart Nginx** so it switches to the HTTPS config (the entrypoint detects `/etc/letsencrypt/live/<first-hostname>/fullchain.pem`):

```bash
docker compose -f docker-compose.prod.yml --env-file .env restart nginx
```

After that, HTTP should redirect to HTTPS (except `/.well-known/acme-challenge/` for renewals).

## 5. Renewals and reloading Nginx

The `certbot` service runs `certbot renew` periodically. After a successful renew, reload Nginx so it picks up the new certificate files (paths unchanged on disk).

From the project root on the **host**, you can use:

```bash
./deploy/renew-and-reload.sh
```

Or add a **cron** job (example: daily at 03:00):

```cron
0 3 * * * cd /path/to/trade-claw && ./deploy/renew-and-reload.sh >>/var/log/trade-claw-certbot.log 2>&1
```

## 6. Operations

- **Logs:** `docker compose -f docker-compose.prod.yml logs -f streamlit` (or `nginx`, `celery-worker`, etc.).
- **Stop:** `docker compose -f docker-compose.prod.yml down`
- **App data** (SQLite, `.kite_session.json`, etc.) lives in the **`trade_claw_app_data`** named volume; TLS material in **`letsencrypt`**; ACME webroot in **`certbot-www`**.

## Kite login behind HTTPS

Streamlit and the worker share **`KITE_SESSION_FILE=/app/data/.kite_session.json`** on the `trade_claw_app_data` volume. Log in once via the Streamlit UI over HTTPS so the worker can use the same token file.
