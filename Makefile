MONITORING_SERVICES = \
  prometheus \
  alertmanager \
  loki \
  promtail \
  grafana \
  cadvisor \
  evidently-api \
  drift-controller

ML_SERVICES = \
  postgres \
  extract_data \
  synthet_data \
  prepare_data \
  import_data \
  train_model \
  mlflow_management 

UI_SERVICES = \
  postgres \
  streamlit \
  agent \
  predict_api \
  auth_api

PROXY_SERVICE = nginx

AIRFLOW_SERVICES = \
  postgres-airflow \
  airflow-init \
  airflow-webserver \
  airflow-scheduler

BLUE = \033[1;34m
GREEN = \033[1;32m
RED = \033[1;31m
YELLOW = \033[1;33m
RESET = \033[0m

### START COMMANDS ###
start-monitoring:
	@echo "$(BLUE)[START] Starting monitoring/observability...$(RESET)"
	docker compose up -d --no-deps $(MONITORING_SERVICES) 

start-ml:
	@echo "$(BLUE)[START] Starting ML services...$(RESET)"
	docker compose up -d $(ML_SERVICES)

start-ui:
	@echo "$(BLUE)[START] Starting user interface (Streamlit + Agent)...$(RESET)"
	docker compose up -d --no-deps $(UI_SERVICES)

start-proxy:
	@echo "$(BLUE)[START] Starting nginx proxy...$(RESET)"
	docker compose up -d --no-deps $(PROXY_SERVICE)

start-airflow:
	@echo "$(BLUE)[START] Starting Airflow...$(RESET)"
	docker compose up -d $(AIRFLOW_SERVICES)

### STOP COMMANDS ###
stop-monitoring:
	@echo "$(RED)[STOP] Stopping monitoring/observability...$(RESET)"
	docker compose stop $(MONITORING_SERVICES)

stop-ml:
	@echo "$(RED)[STOP] Stopping ML services...$(RESET)"
	docker compose stop $(ML_SERVICES)

stop-ui:
	@echo "$(RED)[STOP] Stopping user interface and database...$(RESET)"
	docker compose stop $(UI_SERVICES) postgres

stop-proxy:
	@echo "$(RED)[STOP] Stopping nginx proxy...$(RESET)"
	docker compose stop $(PROXY_SERVICE)

stop-airflow:
	@echo "$(RED)[STOP] Stopping Airflow...$(RESET)"
	docker compose stop $(AIRFLOW_SERVICES)

### RESTART COMMANDS ###

restart-monitoring: stop-monitoring start-monitoring

restart-ml: stop-ml start-ml

restart-ui: stop-ui start-ui

restart-proxy: stop-proxy start-proxy

restart-airflow: stop-airflow start-airflow

### START/STOP ALL ###
start-all: start-ml start-monitoring start-airflow start-ui

stop-all: stop-ui stop-airflow stop-monitoring stop-ml 

restart-all: stop-all start-all

### CLEANUP COMMANDS ###
clean:
	@echo "$(YELLOW)[CLEAN] Stopping containers (without volumes)...$(RESET)"
	docker compose down

purge:
	@echo "$(YELLOW)[PURGE] Stopping containers and removing volumes...$(RESET)"
	docker compose down -v

### HELP ###
help:
	@echo "Available commands:"
	@echo "  start-airflow     : Start Airflow services"
	@echo "  start-ml          : Start ML services"
	@echo "  start-monitoring  : Start monitoring"
	@echo "  start-ui          : Start user interface (Streamlit + agent)"
	@echo "  start-proxy       : Start nginx proxy only"
	@echo ""
	@echo "  stop-airflow      : Stop Airflow services"
	@echo "  stop-ml           : Stop ML services"
	@echo "  stop-monitoring   : Stop monitoring"
	@echo "  stop-ui           : Stop user interface"
	@echo "  stop-proxy        : Stop nginx proxy only"
	@echo ""
	@echo "  restart-airflow   : Restart Airflow services"
	@echo "  restart-ml        : Restart ML services"
	@echo "  restart-monitoring: Restart monitoring"
	@echo "  restart-ui        : Restart user interface"
	@echo "  restart-proxy     : Restart nginx proxy only"
	@echo ""
	@echo "  start-all         : Start all services"
	@echo "  stop-all          : Stop all services"
	@echo "  restart-all       : Restart all services"
	@echo ""
	@echo "  clean             : Stop containers without removing volumes"
	@echo "  purge             : Stop containers and remove volumes"
	@echo ""
	@echo "Usage: make <command>"

