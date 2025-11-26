# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Kimi-K2-Thinking Kubernetes Deployment

This project deploys the Kimi-K2-Thinking model using vLLM on a Kubernetes cluster with 16x H100-80GB GPUs (2 nodes × 8 GPUs each) using Ray for distributed serving.

## Development Setup

### Dependencies
```bash
# Install Python dependencies (requires Python 3.13+)
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Project Structure
```
src/
├── k8s/                           # Kubernetes manifests
│   └── tp8pp2/                    # TP=8, PP=2 configuration (16 GPUs)
│       ├── 00-namespace.yaml      # vllm-kimi namespace
│       ├── 01-ray-head.yaml       # Ray head pod (g327, 8 GPUs)
│       ├── 02-ray-worker.yaml     # Ray worker pod (g328, 8 GPUs)
│       └── 03-service.yaml        # vLLM API service
└── kimi_k2_thinking_k8s/
    └── client.py                  # Test client for vLLM API
```

### Kubernetes Operations

**Deploy the vLLM cluster:**
```bash
# Apply all manifests in order
kubectl apply -f src/k8s/tp8pp2/00-namespace.yaml
kubectl apply -f src/k8s/tp8pp2/01-ray-head.yaml
kubectl apply -f src/k8s/tp8pp2/02-ray-worker.yaml
kubectl apply -f src/k8s/tp8pp2/03-service.yaml

# Or apply all at once
kubectl apply -f src/k8s/tp8pp2/
```

**Monitor deployment:**
```bash
# Check pod status
kubectl get pods -n vllm-kimi

# Check Ray head logs (model loading, vLLM startup)
kubectl logs -n vllm-kimi ray-head -f

# Check Ray worker logs
kubectl logs -n vllm-kimi ray-worker -f

# Check service
kubectl get svc -n vllm-kimi
```

**Testing the deployment:**
```bash
# Port forward the service to localhost
kubectl port-forward -n vllm-kimi svc/vllm-api-service 8000:8000

# Test using the client
python src/kimi_k2_thinking_k8s/client.py --url http://localhost:8000/v1
```

**Cleanup:**
```bash
# Delete all resources
kubectl delete -f src/k8s/tp8pp2/

# Or delete namespace (removes everything)
kubectl delete namespace vllm-kimi
```

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type check
basedpyright

# Run pre-commit on all files
pre-commit run --all-files
```

### Git Commit Convention
This project uses conventional commits enforced by pre-commit hooks:
- `feat:` - New features
- `fix:` - Bug fixes
- `chore:` - Maintenance tasks
- `docs:` - Documentation changes
- `refactor:` - Code refactoring

## Architecture Overview

### Multi-Node Distributed Serving
The deployment uses a **two-node Ray cluster** with pipeline parallelism for high throughput:

**Ray Head (g327):**
- Runs Ray GCS server and vLLM server entrypoint
- Handles API requests on port 8000
- Uses 8 GPUs for tensor parallelism (TP=8)
- Model loading and inference coordination

**Ray Worker (g328):**
- Connects to Ray head via Ray cluster
- Provides 8 additional GPUs for pipeline stage 2
- Handles pipeline-parallel model shards

**Communication:**
- Ray cluster coordination via port 6379
- Inter-GPU communication via NCCL (disabled InfiniBand initially, using Ethernet)
- Host networking enabled for optimal performance

### vLLM Configuration Strategy
The manifests configure vLLM with:
- `--tensor-parallel-size 8`: Split model across 8 GPUs per node
- `--pipeline-parallel-size 2`: Pipeline across 2 nodes (head + worker)
- `--distributed-executor-backend ray`: Use Ray for multi-node coordination
- Model caching on local NVMe (`/data/models/huggingface` on both nodes)

### Critical Design Decisions
1. **Host networking**: Enabled for optimal InfiniBand/RDMA performance
2. **Shared memory**: 100Gi `/dev/shm` for tensor-parallel NCCL operations
3. **Node affinity**: Hard-pinned to g327/g328 for consistent performance
4. **Model caching**: HostPath volumes to 17TB NVMe on each node
5. **Resource limits**: 8 GPUs, 300-500Gi RAM, 32-60 CPUs per pod

## Infrastructure Analysis

### Cluster Overview
**Kubernetes Version:** v1.33.5
**CNI:** Cilium
**Container Runtime:** containerd 1.7.28

### Nodes

#### Control Plane Nodes (3)
- `control-plane-i-00c28e56c448b6b5e` (10.105.23.63)
- `control-plane-i-02b5cb1ed447ec100` (10.105.8.136)
- `control-plane-i-05657673530811362` (10.105.33.180)

#### Worker Nodes (2) - GPU Nodes
**Node: g327**
- Internal IP: 10.15.35.65
- Machine: Dell PowerEdge XE9680
- OS: Ubuntu 22.04.5 LTS
- Kernel: 5.15.0-161-generic
- CPUs: 104 cores
- Memory: ~1TB (1056208624Ki)
- Storage: ~440GB ephemeral + **17TB in /data (aggregated NVMe)**
- **GPUs: 8x NVIDIA H100-80GB-HBM3**
- GPU Memory: 81559 MB per GPU
- Compute Capability: 9.0 (Hopper architecture)
- CUDA Driver: Pre-installed
- RDMA: Capable
- Topology: cluster3/rack5

**Node: g328**
- Internal IP: 10.15.35.73
- Machine: Dell PowerEdge XE9680
- OS: Ubuntu 22.04.5 LTS
- Kernel: 5.15.0-161-generic
- CPUs: 104 cores
- Memory: ~1TB (1056208592Ki)
- Storage: ~440GB ephemeral + **17TB in /data (aggregated NVMe)**
- **GPUs: 8x NVIDIA H100-80GB-HBM3**
- GPU Memory: 81559 MB per GPU
- Compute Capability: 9.0 (Hopper architecture)
- CUDA Driver: Pre-installed
- RDMA: Capable
- Topology: cluster3/rack5

**Total GPU Resources: 16x H100-80GB**

### GPU Infrastructure

#### NVIDIA Device Plugin
✅ **Status:** Running and operational
**DaemonSets in kube-nvidia namespace:**
- `nvidia-device-plugin-daemonset` (2/2 running)
- `nvidia-container-toolkit-daemonset` (2/2 running)
- `nvidia-dcgm-exporter` (2/2 running)
- `gpu-feature-discovery` (2/2 running)
- `nvidia-mig-manager` (2/2 running)
- `nvidia-operator-validator` (2/2 running)

#### GPU Configuration
- **MIG Mode:** Disabled (all-disabled)
- **MPS Mode:** Not capable/not enabled
- **Sharing Strategy:** None (exclusive GPU access)
- **Driver:** Pre-installed (not managed by operator)

### Network Infrastructure
- **RDMA Capable:** Yes (InfiniBand available)
- **Inter-node networking:** High-speed, suitable for distributed training/serving
- **Pod CIDR g327:** 192.168.4.0/24
- **Pod CIDR g328:** 192.168.3.0/24

### Existing Operators/CRDs
- **MPI Operator:** Installed (kubeflow.org/mpijobs)
- **Ray Operator:** Not installed (needs to be added)

### Storage Classes
No storage classes found in cluster.

**Available Storage:**
- **17TB in /data on both nodes** (aggregated NVMe arrays) ⭐
- Ephemeral storage: ~440GB per node
- Strategy: Use hostPath volumes pointing to /data for model caching

### Namespaces
- default
- kube-system
- kube-nvidia
- kube-public
- kube-node-lease
- cilium-secrets
- monitoring
- mpi-operator

## Kimi-K2-Thinking Requirements

### Model Specifications
**Model:** moonshotai/Kimi-K2-Thinking
**Minimum Requirements:** 8x H200/H20 GPUs (we have 16x H100-80GB ✅)

### vLLM Configuration Requirements

#### For Low Latency (Single Node - 8 GPUs)
```bash
vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code
```

#### For High Throughput (Multi-Node - 16 GPUs) ⭐ RECOMMENDED
```bash
vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --decode-context-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code
```

**Performance Gains with DCP:**
- 43.1% faster token throughput
- 25.6% higher request throughput

### Distributed Serving Strategy

**Architecture:** Multi-node with Ray cluster

```
Ray Cluster Layout:
┌─────────────────────────────────────────┐
│  Ray Head Pod (on g327)                 │
│  ├─ Ray head service                    │
│  ├─ vLLM server process                 │
│  ├─ 8x H100 GPUs                        │
│  └─ Resource: nvidia.com/gpu: 8         │
├─────────────────────────────────────────┤
│  Ray Worker Pod (on g328)               │
│  ├─ Ray worker node                     │
│  ├─ vLLM worker process                 │
│  ├─ 8x H100 GPUs                        │
│  └─ Resource: nvidia.com/gpu: 8         │
└─────────────────────────────────────────┘

Configuration:
- tensor_parallel_size=8 (GPUs per node)
- pipeline_parallel_size=2 (number of nodes)
- decode_context_parallel_size=8 (for throughput)
```

### Key Configuration Parameters

**vLLM Flags:**
- `--tensor-parallel-size 8`: Distribute model across 8 GPUs per node
- `--pipeline-parallel-size 2`: Pipeline across 2 nodes
- `--decode-context-parallel-size 8`: Parallel decoding for throughput
- `--enable-auto-tool-choice`: Enable tool calling
- `--tool-call-parser kimi_k2`: Kimi-K2 specific parser
- `--reasoning-parser kimi_k2`: Kimi-K2 reasoning extraction
- `--trust-remote-code`: Required for custom model code

**Resource Requirements:**
- CPU: ~16-32 cores per pod
- Memory: ~200-400GB per pod
- Shared Memory: Large /dev/shm volume (essential for tensor parallel)
- GPU: 8x nvidia.com/gpu per pod

### Deployment Strategy

1. **Install Ray Operator** (if not using raw pods)
   - OR use StatefulSet with Ray init containers

2. **Create Namespace:** `vllm-kimi` or similar

3. **Ray Cluster Setup:**
   - Ray head service for discovery
   - Ray head pod on g327
   - Ray worker pod on g328
   - Both connected via Ray cluster

4. **vLLM Deployment:**
   - Deploy as part of Ray pods
   - Mount /dev/shm as emptyDir with large size
   - Configure node affinity to pin pods to g327/g328
   - Set up liveness/readiness probes

5. **Service Exposure:**
   - ClusterIP service for internal access
   - LoadBalancer or NodePort for external access
   - Health endpoint: /health
   - API endpoints: /v1/completions, /v1/chat/completions

### Critical Kubernetes Configurations

**Shared Memory:**
```yaml
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 100Gi
```

**GPU Resource Request:**
```yaml
resources:
  limits:
    nvidia.com/gpu: 8
  requests:
    nvidia.com/gpu: 8
```

**Node Affinity:**
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - g327  # or g328
```

### Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 300
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
```

## Network and Security Considerations

### RDMA Benefits
- Ultra-low latency for inter-GPU communication
- Critical for pipeline parallelism across nodes
- Infinband available on both nodes

### Security
- No MIG isolation configured (full GPU access)
- Pods will have exclusive GPU access
- Consider network policies for production

## Next Steps

1. ✅ Infrastructure analysis complete
2. 📝 Create namespace manifest
3. 📝 Create Ray cluster manifests (head + worker)
4. 📝 Create vLLM StatefulSet/Deployment
5. 📝 Create Services (Ray cluster service + vLLM API service)
6. 📝 Create monitoring/observability configs (optional)
7. 🚀 Deploy to cluster
8. ✅ Test with sample requests

## Model Storage Strategy

**Selected Strategy: HostPath on /data** ⭐

Both g327 and g328 have 17TB aggregated NVMe storage mounted at /data.

**Implementation:**
```yaml
volumes:
  - name: model-cache
    hostPath:
      path: /data/models/huggingface
      type: DirectoryOrCreate
```

**Benefits:**
- 17TB available space (plenty for models)
- Fast NVMe performance
- Persistent across pod restarts
- No network overhead
- Models cached locally on each node

**Model Paths:**
- g327: `/data/models/huggingface` → mounted to `/root/.cache/huggingface` in pod
- g328: `/data/models/huggingface` → mounted to `/root/.cache/huggingface` in pod

**Note:** Model will be downloaded once per node (~100GB+ for Kimi-K2-Thinking), then cached for subsequent pod restarts.

## Estimated Resource Usage

**Per Pod:**
- GPUs: 8x H100-80GB
- CPU: ~20-30 cores (requests), 50+ for limits
- Memory: ~300GB (model weights + KV cache + activations)
- Storage: ~150GB (model + cache)
- Shared Memory: ~50-100GB

**Total Cluster:**
- 16 GPUs (fully utilized)
- ~40-60 CPU cores
- ~600GB RAM
- ~300GB storage
