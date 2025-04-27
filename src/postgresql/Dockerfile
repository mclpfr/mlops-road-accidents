FROM postgres:15-alpine

# Configuration des limites de mémoire
ENV POSTGRES_SHARED_BUFFERS=128MB
ENV POSTGRES_EFFECTIVE_CACHE_SIZE=256MB
ENV POSTGRES_WORK_MEM=16MB
ENV POSTGRES_MAINTENANCE_WORK_MEM=64MB

COPY src/postgresql/init-db.sh /docker-entrypoint-initdb.d/

RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh
