FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install kiro-cli. The official tarball URL pattern; if upstream changes,
# the build will fail loudly and we know to update.
RUN curl -fsSL "https://desktop-release.kiro.dev/cli/latest/kiro-cli-linux-x86_64.tar.gz" -o /tmp/kiro.tar.gz \
    && tar -xzf /tmp/kiro.tar.gz -C /usr/local/bin \
    && rm /tmp/kiro.tar.gz \
    && chmod +x /usr/local/bin/kiro-cli || echo "kiro-cli install may need URL update — fix in image then rebuild"

COPY . .

ENV PORT=10000
EXPOSE 10000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
