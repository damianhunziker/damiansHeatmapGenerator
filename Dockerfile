FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

ARG DEV_UID=1000
ARG DEV_GID=1000
ARG DEV_USER=dev

RUN groupadd --gid "${DEV_GID}" "${DEV_USER}" \
    && useradd --uid "${DEV_UID}" --gid "${DEV_GID}" \
        --create-home --shell /bin/bash "${DEV_USER}"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p ohlc_cache data_cache html_cache pnl_cache automator_html \
    && chown -R "${DEV_UID}:${DEV_GID}" /app

USER ${DEV_USER}

# Static HTML viewer for generated reports (see docker-compose.yml "viewer" service)
EXPOSE 8900

CMD ["sleep", "infinity"]
