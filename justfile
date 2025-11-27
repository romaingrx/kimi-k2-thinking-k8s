port_forward_services:
    @echo "Starting port forwards..."
    @kubectl port-forward -n monitoring svc/jaeger-query 16686:16686 & echo $$! > /tmp/pf1.pid; \
    kubectl port-forward -n monitoring svc/prometheus 9090:9090 & echo $$! > /tmp/pf2.pid; \
    kubectl port-forward -n monitoring svc/grafana 3000:3000 & echo $$! > /tmp/pf3.pid; \
    kubectl port-forward -n monitoring svc/alertmanager 9093:9093 & echo $$! > /tmp/pf4.pid; \
    kubectl port-forward -n vllm-kimi svc/vllm-api 8000:8000 & echo $$! > /tmp/pf5.pid; \
    trap "kill \$$(cat /tmp/pf*.pid) 2>/dev/null; rm -f /tmp/pf*.pid" EXIT INT TERM; \
    echo "Port forwards active. Press Ctrl+C to stop."; \
    wait
