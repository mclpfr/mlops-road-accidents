FROM nginx:latest

# Remove the default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom configuration
COPY config/default.conf /etc/nginx/conf.d/default.conf
