FROM postgres:15-alpine

# Install yq, a lightweight command-line YAML processor.
# We need this to parse the config.yaml file.
RUN apk --no-cache add yq

# Copy the wrapper script into the container.
COPY src/postgresql/entrypoint-wrapper.sh /usr/local/bin/entrypoint-wrapper.sh

# Ensure the script is executable.
RUN chmod +x /usr/local/bin/entrypoint-wrapper.sh

# Set our wrapper script as the new entrypoint.
ENTRYPOINT ["entrypoint-wrapper.sh"]

# The default command for the postgres image is 'postgres'.
# Our wrapper script will call the original entrypoint, which in turn calls this command.
CMD ["postgres"]

# Configuration des limites de mémoire
ENV POSTGRES_SHARED_BUFFERS=128MB
ENV POSTGRES_EFFECTIVE_CACHE_SIZE=256MB
ENV POSTGRES_WORK_MEM=16MB
ENV POSTGRES_MAINTENANCE_WORK_MEM=64MB

COPY src/postgresql/init-db.sh /docker-entrypoint-initdb.d/

RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh
