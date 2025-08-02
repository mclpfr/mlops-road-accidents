# Use the official Grafana image as a base
FROM grafana/grafana:latest

# Switch to root user to install packages
USER root

# Install yq and other dependencies
RUN apk --no-cache add yq

# Copy the custom entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh



# Set the custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]
