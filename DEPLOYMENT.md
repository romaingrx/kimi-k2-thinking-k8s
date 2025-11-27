# vLLM Kimi-K2-Thinking Deployment Guide

Complete guide for deploying vLLM with InfiniBand and tensor core monitoring on Kubernetes.

## Prerequisites

- Kubernetes cluster with 2 GPU nodes (g258, g268)
- 16x H100-80GB GPUs total (8 per node)
- InfiniBand network configured (mlx5 interfaces)
- NVIDIA GPU Operator installed

## Quick Start (Full Deployment)

```bash
# 1. Deploy monitoring stack (Prometheus, Grafana, Jaeger, DCGM)
kubectl apply -f src/k8s/monitoring/

# 2. Configure DCGM exporter for tensor core metrics
kubectl patch daemonset nvidia-dcgm-exporter -n kube-nvidia --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/args", "value": ["-f", "/etc/dcgm-exporter/metrics.csv"]},
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts/-", "value": {"name": "metrics-config", "mountPath": "/etc/dcgm-exporter", "readOnly": true}},
  {"op": "add", "path": "/spec/template/spec/volumes/-", "value": {"name": "metrics-config", "configMap": {"name": "dcgm-exporter-metrics"}}}
]'

# 3. Deploy vLLM cluster with InfiniBand enabled
kubectl apply -f src/k8s/tp16/

# 4. Monitor deployment
kubectl logs -n vllm-kimi ray-head -f
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Node g258 (8x H100)          Node g268 (8x H100)  │
│  ┌────────────────┐            ┌────────────────┐  │
│  │   ray-head     │◄──IB────►│  ray-worker    │  │
│  │   TP=16 leader │   RDMA    │   TP=16 node2  │  │
│  │   vLLM server  │            │   vLLM worker  │  │
│  └────────────────┘            └────────────────┘  │
│         ▲                              ▲            │
│         │                              │            │
│         └──────NCCL All-Reduce─────────┘            │
│                (via InfiniBand)                     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Monitoring Stack                       │
│  ┌───────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ │
│  │Prometheus │ │ Grafana  │ │ Jaeger │ │  DCGM  │ │
│  └───────────┘ └──────────┘ └────────┘ └────────┘ │
└─────────────────────────────────────────────────────┘
```

## Key Configuration Changes

### 1. InfiniBand with GPUDirect RDMA

**Files Modified:**

- `src/k8s/tp16/01-ray-head.yaml` (lines 103-119)
- `src/k8s/tp16/02-ray-worker.yaml` (lines 67-83)

**Environment Variables Added:**

```yaml
- name: NCCL_NET
  value: "IB" # Force InfiniBand
- name: NCCL_IB_HCA
  value: "mlx5" # Mellanox adapter
- name: NCCL_IB_GID_INDEX
  value: "3" # RoCE v2
- name: NCCL_NET_GDR_LEVEL
  value: "5" # Enable GPUDirect RDMA
- name: NCCL_IB_QPS_PER_CONNECTION
  value: "4" # 4 queue pairs for bandwidth
```

**Expected Result:**

- NCCL logs should show: `[send] via NET/IB/GDRDMA` ✅
- NOT: `[send] via NET/Socket` ❌

### 2. Tensor Core Profiling Metrics

**Files Added:**

- `src/k8s/monitoring/10-dcgm-metrics.yaml` - ConfigMap with profiling metrics

**Key Metrics Enabled:**

- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` - **Actual tensor core utilization**
- `DCGM_FI_PROF_SM_ACTIVE` - SM activity (what most dashboards show)
- `DCGM_FI_PROF_DRAM_ACTIVE` - Memory bandwidth
- `DCGM_FI_PROF_PCIE_TX/RX_BYTES` - PCIe traffic

## Deployment Steps (Detailed)

### Step 1: Deploy Monitoring Stack

```bash
cd /path/to/kimi-k2-thinking-k8s

# Apply all monitoring resources
kubectl apply -f src/k8s/monitoring/

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=120s
kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=120s
kubectl wait --for=condition=ready pod -l app=jaeger -n monitoring --timeout=120s
```

### Step 2: Enable DCGM Profiling Metrics

```bash
# ConfigMap is already created from step 1
# Now patch the DCGM exporter DaemonSet
kubectl patch daemonset nvidia-dcgm-exporter -n kube-nvidia --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args",
    "value": ["-f", "/etc/dcgm-exporter/metrics.csv"]
  },
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/volumeMounts/-",
    "value": {
      "name": "metrics-config",
      "mountPath": "/etc/dcgm-exporter",
      "readOnly": true
    }
  },
  {
    "op": "add",
    "path": "/spec/template/spec/volumes/-",
    "value": {
      "name": "metrics-config",
      "configMap": {
        "name": "dcgm-exporter-metrics"
      }
    }
  }
]'

# Wait for DaemonSet rollout
kubectl rollout status daemonset nvidia-dcgm-exporter -n kube-nvidia --timeout=300s

# Verify tensor core metrics are available
kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -- \
  curl -s http://nvidia-dcgm-exporter.kube-nvidia.svc.cluster.local:9400/metrics | \
  grep DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
```

### Step 3: Deploy vLLM Cluster

```bash
# Deploy namespace, ray-head, ray-worker, and service
kubectl apply -f src/k8s/tp16/

# Monitor ray-head startup (takes 3-5 minutes)
kubectl logs -n vllm-kimi ray-head -f

# Watch for these key log lines:
# 1. "Loading safetensors checkpoint shards: 100%"
# 2. "[send] via NET/IB/GDRDMA" (InfiniBand active!)
# 3. "vLLM API server running"
```

### Step 4: Verify InfiniBand is Active

```bash
# Check NCCL logs for InfiniBand usage
kubectl logs -n vllm-kimi ray-head | grep "NCCL INFO" | grep -E "IB|Socket" | head -20

# Should see:
# ✅ "via NET/IB/GDRDMA" or "via NET/IB/0"
# ❌ NOT "via NET/Socket/0"
```

### Step 5: Setup Grafana Dashboard

```bash
# Port forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000 &

# Open http://localhost:3000
# Login: admin/admin

# Import NVIDIA DCGM Dashboard:
# 1. Go to Dashboards → Import
# 2. Enter dashboard ID: 23382
# 3. Select "Prometheus" as datasource
# 4. Import
```

## Verification & Testing

### 1. Check vLLM API Health

```bash
kubectl port-forward -n vllm-kimi svc/vllm-api 8000:8000 &

# Test API
curl http://localhost:8000/health

# Send test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Thinking",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 2. Load Test with Tensor Core Monitoring

```bash
# Terminal 1: Monitor GPUs in real-time
./src/kimi_k2_thinking_k8s/profile_gpu.sh

# Terminal 2: Run stress test
python src/kimi_k2_thinking_k8s/stress_test.py \
  --url http://localhost:8000/v1 \
  -w 50 \
  -d 120

# Terminal 3: Watch Grafana dashboard
# Look for:
# - DCGM_FI_PROF_PIPE_TENSOR_ACTIVE rising to 60-80%
# - Power draw increasing to 400-600W
# - Temperature rising to 55-70°C
```

### 3. Expected Performance Metrics

With proper InfiniBand and load:

| Metric                      | Expected Value   |
| --------------------------- | ---------------- |
| **Tensor Core Utilization** | 60-80%           |
| **GPU Utilization**         | 90-100%          |
| **Memory Utilization**      | 40-60%           |
| **Power per GPU**           | 400-600W         |
| **Temperature**             | 55-70°C          |
| **Concurrent Requests**     | 50-100           |
| **Generation Throughput**   | 300-500 tokens/s |

## Troubleshooting

### InfiniBand Not Working

**Symptom:** Logs show `[send] via NET/Socket/0`

**Diagnosis:**

```bash
# Check InfiniBand devices on nodes
kubectl exec -n vllm-kimi ray-head -- ls /dev/infiniband/
# Should see: uverbs0, uverbs1, etc.

# Check Mellanox drivers
kubectl exec -n vllm-kimi ray-head -- ibstat
```

**Fix:**

- Verify InfiniBand is physically connected
- Check NCCL_IB_HCA matches your adapter (mlx5_0, mlx5_1, etc.)
- Try different NCCL_IB_GID_INDEX values (0, 1, 2, 3)

### Tensor Core Metrics Not Showing

**Symptom:** DCGM_FI_PROF_PIPE_TENSOR_ACTIVE = 0 or missing

**Diagnosis:**

```bash
# Check if profiling metrics are enabled
kubectl get configmap dcgm-exporter-metrics -n kube-nvidia

# Check DCGM exporter configuration
kubectl get daemonset nvidia-dcgm-exporter -n kube-nvidia -o yaml | \
  grep -A 5 "metrics-config"
```

**Fix:**

```bash
# Reapply the DCGM patch from Step 2
# Or restart DCGM exporter pods
kubectl delete pods -n kube-nvidia -l app=nvidia-dcgm-exporter
```

### Low GPU Power (<200W)

**Symptom:** Power stays at 100-200W, tensor cores at 0%

**Diagnosis:**

- Check concurrent load: `kubectl logs -n vllm-kimi ray-head | grep "Running:" | tail -5`
- Need 50+ concurrent requests to saturate GPUs

**Fix:**

```bash
# Run proper concurrent load test
python src/kimi_k2_thinking_k8s/stress_test.py -w 100 -d 120
```

## Performance Tuning

### For Maximum Throughput

```yaml
# In vllm serve command:
--max-num-seqs 1024                  # Handle more concurrent requests
--max-num-batched-tokens 262144      # Larger batches
--num-scheduler-steps 10             # Batch scheduling decisions
```

### For Lower Latency

```yaml
--max-num-seqs 256                   # Fewer concurrent requests
--max-num-batched-tokens 65536       # Smaller batches
--num-scheduler-steps 4              # More frequent scheduling
```

## Cluster Teardown

```bash
# Delete vLLM cluster
kubectl delete -f src/k8s/tp16/

# Delete monitoring stack
kubectl delete -f src/k8s/monitoring/

# Revert DCGM exporter (optional)
kubectl rollout undo daemonset nvidia-dcgm-exporter -n kube-nvidia
```

## References

- [vLLM Distributed Serving](https://docs.vllm.ai/en/stable/serving/distributed_serving/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [DCGM Field IDs](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)
- [Grafana DCGM Dashboard 23382](https://grafana.com/grafana/dashboards/23382)
