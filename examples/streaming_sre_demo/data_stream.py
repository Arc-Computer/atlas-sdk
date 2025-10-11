"""Synthetic incident stream for the streaming SRE continual-learning demo.

This module exposes an async generator that produces Datadog-style JSON log
bundles representing individual incidents. It also provides a CLI utility to
mirror the raw log stream in a terminal while the demo driver runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Iterable, Iterator, Sequence

from rich.console import Console
from rich.table import Table

TENANT_ID = "sre-demo"


@dataclass(frozen=True)
class LogTemplate:
    """Template describing an individual log entry."""

    message: str
    source: str = "kubernetes"
    status: str = "info"
    delay_seconds: float = 1.0
    tags: Sequence[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IncidentDefinition:
    """Definition for a synthetic incident."""

    incident_id: int
    incident_type: str
    severity: str
    service: str
    host: str
    ground_truth: str
    logs: Sequence[LogTemplate]
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: Sequence[dict[str, Any]] = field(default_factory=tuple)
    recent_changes: Sequence[dict[str, Any]] = field(default_factory=tuple)
    related_incidents: Sequence[dict[str, Any]] = field(default_factory=tuple)
    runbook_hint: str | None = None


def _format_ddtags(tags: Sequence[str] | None) -> str:
    if not tags:
        return ""
    return ",".join(tag.strip() for tag in tags if tag)


def _default_incident_plan() -> list[IncidentDefinition]:
    logs = {
        "cpu_throttle": [
            LogTemplate(
                message="Kubelet observed sustained CPU throttling beyond 85% for deployment api-gateway",
                status="warning",
                extra={"metric": "cpu_throttled_seconds_total", "value": 0.86},
            ),
            LogTemplate(
                message="HorizontalPodAutoscaler insufficient replicas: desired 12, current 6. Throttling persists.",
                source="kubernetes",
                status="error",
            ),
            LogTemplate(
                message="SLO breach predicted: p99 latency projected to exceed 450ms within 5 minutes.",
                source="monitoring",
                status="critical",
                extra={"predicted_latency_ms": 470},
            ),
        ],
        "cache_eviction": [
            LogTemplate(
                message="Redis eviction policy triggered for namespace cart-cache (evicted_keys=124 in 60s)",
                source="redis",
                status="error",
            ),
            LogTemplate(
                message="Downstream checkout-service reports cache miss spike (miss_ratio=0.61)",
                source="checkout-service",
                status="warning",
            ),
            LogTemplate(
                message="Queue depth increasing on checkout-service (pending_jobs=487)",
                source="worker",
                status="warning",
            ),
        ],
        "crash_loop": [
            LogTemplate(
                message="Container web-frontend terminated with exit code 139 (segfault) after 12s uptime",
                status="critical",
            ),
            LogTemplate(
                message="Readiness probe failed: /healthz returned 500 (panic: nil pointer dereference)",
                status="critical",
            ),
            LogTemplate(
                message="Kubelet set CrashLoopBackOff for pod web-frontend-7f97c9f8df-pt5lz (restart_count=5)",
                status="critical",
            ),
        ],
        "image_pull": [
            LogTemplate(
                message="ImagePullBackOff: Failed to pull image registry.prod.arc/api-gateway:2025.10.09",
                status="error",
            ),
            LogTemplate(
                message="Registry response 403: signature mismatch for digest sha256:e8ae...",
                source="registry",
                status="error",
            ),
            LogTemplate(
                message="Deployment rollout paused awaiting valid image signature approval",
                source="kubernetes",
                status="warning",
            ),
        ],
        "novel_mtls": [
            LogTemplate(
                message="Envoy downstream TLS handshake failed: downstream_cert_error (error_detail: CERTIFICATE_EXPIRED)",
                source="service-mesh",
                status="critical",
                extra={"handshake_error": "CERTIFICATE_EXPIRED"},
            ),
            LogTemplate(
                message="gRPC dial failure from service api-gateway to payment-router: remote error tls: unknown certificate",
                source="api-gateway",
                status="critical",
            ),
            LogTemplate(
                message="Mutual TLS policy enforcement rejected connection on port 8443 (spiffe://payments.router mismatch)",
                source="service-mesh",
                status="critical",
            ),
            LogTemplate(
                message="On-call guidance: rotate client certificate bundle; scope payments-router mTLS profiles",
                source="runbook",
                status="info",
            ),
        ],
        "config_drift": [
            LogTemplate(
                message="Detected drift: configmap feature-flags missing key payment_fallback (hash mismatch)",
                source="config-controller",
                status="warning",
            ),
            LogTemplate(
                message="New pods launched with outdated env var PAYMENT_GATEWAY_URL pointing to staging endpoint",
                source="kubernetes",
                status="error",
            ),
            LogTemplate(
                message="Payment error rate climbed to 12% (baseline 0.2%) within 4 minutes",
                source="monitoring",
                status="critical",
            ),
        ],
        "network_partition": [
            LogTemplate(
                message="Calico controller detected BGP session flap between edge node ewr-az2 and spine-12",
                source="network",
                status="error",
            ),
            LogTemplate(
                message="95th percentile packet loss 27% between api-gateway and payments-router (link_id=ewr-spine-12)",
                source="network",
                status="critical",
            ),
            LogTemplate(
                message="Service mesh rerouting 38% of traffic via backup path (latency penalty +180ms)",
                source="service-mesh",
                status="warning",
            ),
        ],
        "db_pool": [
            LogTemplate(
                message="pgbouncer connection limit reached: active=400, max_client_conn=400",
                source="database",
                status="critical",
            ),
            LogTemplate(
                message="Checkout service threads blocked waiting for DB connections (avg_wait_ms=2100)",
                source="checkout-service",
                status="critical",
            ),
            LogTemplate(
                message="Slow query backlog > 120 (statement: SELECT * FROM orders WHERE status='pending')",
                source="database",
                status="warning",
            ),
        ],
        "readiness_probe": [
            LogTemplate(
                message="Readiness probe failing for 2 consecutive minutes on pod api-gateway-7fc5d8c65c-mb8k2",
                status="error",
            ),
            LogTemplate(
                message="Downstream dependency payment-router returned 503 (circuit breaker open)",
                source="api-gateway",
                status="warning",
            ),
            LogTemplate(
                message="HPA scaled replicas from 6→12 but readiness remains false",
                source="kubernetes",
                status="warning",
            ),
        ],
        "timeout_spike": [
            LogTemplate(
                message="p95 request latency jumped to 920ms (baseline 210ms) for endpoint POST /checkout",
                source="api-gateway",
                status="warning",
            ),
            LogTemplate(
                message="TimeoutError: payment-router exceeded deadline 400ms (retries=3)",
                source="checkout-service",
                status="error",
            ),
            LogTemplate(
                message="Circuit breaker open on checkout-service -> payment-router path",
                source="service-mesh",
                status="error",
            ),
        ],
    }

    defn = [
        IncidentDefinition(
            incident_id=1,
            incident_type="cpu_throttle",
            severity="warning",
            service="api-gateway",
            host="ip-10-0-7-21",
            ground_truth="cpu_hpa_misconfigured",
            logs=logs["cpu_throttle"],
            tags=("env:prod", "service:api-gateway", "team:sre"),
            metadata={"customer_tier": "enterprise"},
            metrics=(
                {"metric": "kubernetes.cpu.throttled_pct", "value": 0.86, "baseline": 0.35},
                {"metric": "http.p99_latency_ms", "value": 420, "baseline": 180},
            ),
            recent_changes=(
                {"change_id": "deploy_27341", "author": "lisa.chen", "timestamp": "2025-10-11T13:58:00Z", "summary": "Increased rate limiter burst to 5x"},
            ),
            related_incidents=(
                {"incident_id": "INC-3214", "date": "2025-09-17", "summary": "HPA misconfiguration caused sustained throttling"},
            ),
            runbook_hint="Validate HPA targets and compare desired vs. current replicas; inspect recent deploys for CPU-heavy features.",
        ),
        IncidentDefinition(
            incident_id=2,
            incident_type="cache_eviction",
            severity="major",
            service="checkout-service",
            host="ip-10-0-5-18",
            ground_truth="cache_capacity_regression",
            logs=logs["cache_eviction"],
            tags=("env:prod", "service:checkout-service", "component:redis"),
            metrics=(
                {"metric": "redis.evictions_per_min", "value": 124, "baseline": 5},
                {"metric": "checkout.cache_hit_ratio", "value": 0.39, "baseline": 0.92},
            ),
            recent_changes=(
                {"change_id": "config_redis_817", "author": "samir.raza", "timestamp": "2025-10-11T13:40:00Z", "summary": "Reduced maxmemory-policy samples to 3"},
            ),
            related_incidents=(
                {"incident_id": "INC-3098", "date": "2025-08-02", "summary": "Redis eviction storm after ttl change"},
            ),
            runbook_hint="Inspect Redis memory policy and cache size; confirm shopping cart TTL matches capacity expectations.",
        ),
        IncidentDefinition(
            incident_id=3,
            incident_type="crash_loop",
            severity="critical",
            service="web-frontend",
            host="ip-10-0-12-4",
            ground_truth="golang_panic_upgrade",
            logs=logs["crash_loop"],
            tags=("env:prod", "service:web-frontend", "component:k8s"),
            metrics=(
                {"metric": "kubernetes.pod_restarts", "value": 12, "baseline": 1},
                {"metric": "frontend.error_rate_pct", "value": 18.0, "baseline": 0.5},
            ),
            recent_changes=(
                {"change_id": "deploy_frontend_664", "author": "alex.ng", "timestamp": "2025-10-11T13:20:00Z", "summary": "Go 1.23 upgrade + new image"},
            ),
            related_incidents=(
                {"incident_id": "INC-2774", "date": "2025-06-12", "summary": "Segfault due to nil pointer after auth refactor"},
            ),
            runbook_hint="Roll back to previous frontend image; inspect panic stack trace for nil pointer introduced in Go upgrade.",
        ),
        IncidentDefinition(
            incident_id=4,
            incident_type="image_pull_backoff",
            severity="major",
            service="api-gateway",
            host="ip-10-0-7-21",
            ground_truth="registry_signature_mismatch",
            logs=logs["image_pull"],
            tags=("env:prod", "service:api-gateway", "component:registry"),
            metrics=(
                {"metric": "kubernetes.deploy_blocked_replicas", "value": 12, "baseline": 0},
            ),
            recent_changes=(
                {"change_id": "security_policy_192", "author": "maria.garcia", "timestamp": "2025-10-11T13:05:00Z", "summary": "Enabled strict signature verification"},
            ),
            related_incidents=(
                {"incident_id": "INC-2881", "date": "2025-07-22", "summary": "Failed deploy due to mismatched cosign signature"},
            ),
            runbook_hint="Re-sign the container image with production key or revert signature policy change.",
        ),
        IncidentDefinition(
            incident_id=5,
            incident_type="novel_mtls",
            severity="critical",
            service="api-gateway",
            host="ip-10-0-4-12",
            ground_truth="mtls_cert_mismatch",
            logs=logs["novel_mtls"],
            tags=("env:prod", "service:api-gateway", "tls", "region:us-east-1"),
            metadata={"novel": True},
            metrics=(
                {"metric": "envoy.tls.handshake_failures", "value": 58, "baseline": 1},
                {"metric": "http.error_rate_pct", "value": 21.0, "baseline": 0.7},
            ),
            recent_changes=(
                {"change_id": "cert_rotation_1129", "author": "noah.stevens", "timestamp": "2025-10-11T12:55:00Z", "summary": "Rotated payment-router client cert bundle"},
            ),
            related_incidents=(
                {"incident_id": "INC-3301", "date": "2025-09-03", "summary": "mTLS chain mismatch after partial cert rotation"},
            ),
            runbook_hint="Validate SPIFFE IDs on api-gateway clients; ensure payment-router bundle includes new intermediate CA.",
        ),
        IncidentDefinition(
            incident_id=6,
            incident_type="cpu_throttle",
            severity="warning",
            service="etl-worker",
            host="ip-10-0-23-9",
            ground_truth="batch_job_overlap",
            logs=logs["cpu_throttle"],
            tags=("env:prod", "service:etl-worker", "team:data-platform"),
            metrics=(
                {"metric": "etl.cpu.utilization_pct", "value": 95, "baseline": 65},
            ),
            recent_changes=(
                {"change_id": "scheduler_patch_772", "author": "divya.iyer", "timestamp": "2025-10-11T12:30:00Z", "summary": "Reduced cron spacing between ETL batches"},
            ),
            related_incidents=(
                {"incident_id": "INC-3120", "date": "2025-08-28", "summary": "Concurrent ETL runs exhausting CPU quota"},
            ),
            runbook_hint="Stagger ETL start times or raise CPU limits; confirm autoscaler catches backfill spikes.",
        ),
        IncidentDefinition(
            incident_id=7,
            incident_type="config_drift",
            severity="major",
            service="payment-router",
            host="ip-10-0-9-14",
            ground_truth="configmap_missing_flag",
            logs=logs["config_drift"],
            tags=("env:prod", "service:payment-router", "team:payments"),
            metrics=(
                {"metric": "payments.error_ratio", "value": 12.0, "baseline": 0.2},
            ),
            recent_changes=(
                {"change_id": "feature_flag_sync_998", "author": "keiko.sato", "timestamp": "2025-10-11T12:10:00Z", "summary": "Rolled back feature flag after B testing"},
            ),
            related_incidents=(
                {"incident_id": "INC-3011", "date": "2025-08-10", "summary": "Staging URL leak due to config drift"},
            ),
            runbook_hint="Redeploy configmap with canonical feature flags; run drift detector before re-enabling traffic.",
        ),
        IncidentDefinition(
            incident_id=8,
            incident_type="network_partition",
            severity="critical",
            service="api-gateway",
            host="ip-10-0-7-21",
            ground_truth="bgp_flap_edge",
            logs=logs["network_partition"],
            tags=("env:prod", "service:api-gateway", "network"),
            metrics=(
                {"metric": "network.packet_loss_pct", "value": 27, "baseline": 1},
                {"metric": "service-mesh.rerouted_traffic_pct", "value": 38, "baseline": 5},
            ),
            recent_changes=(
                {"change_id": "bgp_peer_update_441", "author": "matt.rogers", "timestamp": "2025-10-11T11:58:00Z", "summary": "Added new edge peer spine-12"},
            ),
            related_incidents=(
                {"incident_id": "INC-2950", "date": "2025-07-02", "summary": "BGP flap affecting east coast ingress"},
            ),
            runbook_hint="Stabilise BGP session on spine-12; consider draining traffic to backup AZ until packet loss clears.",
        ),
        IncidentDefinition(
            incident_id=9,
            incident_type="db_connection_pool",
            severity="major",
            service="checkout-service",
            host="ip-10-0-5-18",
            ground_truth="pgbouncer_pool_exhaustion",
            logs=logs["db_pool"],
            tags=("env:prod", "service:checkout-service", "component:postgres"),
            metrics=(
                {"metric": "pgbouncer.wait_clients", "value": 210, "baseline": 12},
                {"metric": "db.connections_active", "value": 400, "baseline": 180},
            ),
            recent_changes=(
                {"change_id": "pool_config_551", "author": "amir.haddad", "timestamp": "2025-10-11T11:45:00Z", "summary": "Lowered max_client_conn from 600 to 400"},
            ),
            related_incidents=(
                {"incident_id": "INC-3142", "date": "2025-09-05", "summary": "Checkout queue backlog due to pool exhaustion"},
            ),
            runbook_hint="Raise pgbouncer limits or scale out checkout replicas; detect unbounded transactions exceeding 2s.",
        ),
        IncidentDefinition(
            incident_id=10,
            incident_type="cache_eviction",
            severity="minor",
            service="inventory-service",
            host="ip-10-0-3-11",
            ground_truth="redis_ttl_misconfiguration",
            logs=logs["cache_eviction"],
            tags=("env:prod", "service:inventory-service", "component:redis"),
            metrics=(
                {"metric": "inventory.cache_hit_ratio", "value": 0.55, "baseline": 0.9},
            ),
            recent_changes=(
                {"change_id": "ttl_patch_221", "author": "nadia.gao", "timestamp": "2025-10-11T11:20:00Z", "summary": "Increased TTL for inventory keys to 6h"},
            ),
            related_incidents=(
                {"incident_id": "INC-2850", "date": "2025-07-01", "summary": "Inventory TTL mis-set causing stale stock counts"},
            ),
            runbook_hint="Audit TTL configuration; confirm eviction policy suits new key size distribution.",
        ),
        IncidentDefinition(
            incident_id=11,
            incident_type="readiness_probe",
            severity="major",
            service="api-gateway",
            host="ip-10-0-8-6",
            ground_truth="dependency_circuit_breaker",
            logs=logs["readiness_probe"],
            tags=("env:prod", "service:api-gateway", "component:k8s"),
            metrics=(
                {"metric": "api-gateway.readiness_false_pct", "value": 65, "baseline": 2},
                {"metric": "payment-router.503_rate_pct", "value": 18, "baseline": 0.5},
            ),
            recent_changes=(
                {"change_id": "circuit_breaker_tweak_332", "author": "leah.singh", "timestamp": "2025-10-11T11:05:00Z", "summary": "Raised failure threshold for payment-router circuit breaker"},
            ),
            related_incidents=(
                {"incident_id": "INC-3188", "date": "2025-09-12", "summary": "Payment dependency causing readiness probe cascade"},
            ),
            runbook_hint="Investigate downstream payment-router availability; temporarily disable strict readiness gating if needed.",
        ),
        IncidentDefinition(
            incident_id=12,
            incident_type="timeout_spike",
            severity="major",
            service="checkout-service",
            host="ip-10-0-5-18",
            ground_truth="dependency_satellite_latency",
            logs=logs["timeout_spike"],
            tags=("env:prod", "service:checkout-service", "region:us-east-1"),
            metrics=(
                {"metric": "checkout.timeout_rate_pct", "value": 33, "baseline": 2},
                {"metric": "satellite.latency_ms", "value": 480, "baseline": 120},
            ),
            recent_changes=(
                {"change_id": "satellite_route_88", "author": "brooklyn.james", "timestamp": "2025-10-11T10:58:00Z", "summary": "Shifted satellite traffic to new POP"},
            ),
            related_incidents=(
                {"incident_id": "INC-3220", "date": "2025-09-19", "summary": "Satellite region path change increasing latency"},
            ),
            runbook_hint="Verify satellite POP latency; reroute checkout critical path to stable region.",
        ),
        IncidentDefinition(
            incident_id=13,
            incident_type="cpu_throttle",
            severity="warning",
            service="api-gateway",
            host="ip-10-0-7-21",
            ground_truth="cpu_spike_garbage_collection",
            logs=logs["cpu_throttle"],
            tags=("env:prod", "service:api-gateway", "component:k8s"),
            metrics=(
                {"metric": "go.gc_pause_ms", "value": 210, "baseline": 35},
            ),
            recent_changes=(
                {"change_id": "memory_pool_patch_452", "author": "jensen.park", "timestamp": "2025-10-11T10:30:00Z", "summary": "Increased Go heap target to 1.2 GB"},
            ),
            related_incidents=(
                {"incident_id": "INC-2805", "date": "2025-06-28", "summary": "GC regression after enabling request tracing"},
            ),
            runbook_hint="Inspect GC stats and adjust GOGC; ensure trace sampling not locking main goroutines.",
        ),
        IncidentDefinition(
            incident_id=14,
            incident_type="cache_eviction",
            severity="major",
            service="checkout-service",
            host="ip-10-0-5-18",
            ground_truth="redis_cluster_resync",
            logs=logs["cache_eviction"],
            tags=("env:prod", "service:checkout-service", "component:redis"),
            metrics=(
                {"metric": "redis.repl_backlog_bytes", "value": 524288000, "baseline": 10485760},
            ),
            recent_changes=(
                {"change_id": "redis_cluster_failover_45", "author": "mateo.rojas", "timestamp": "2025-10-11T10:05:00Z", "summary": "Failover shard 3 to new primary"},
            ),
            related_incidents=(
                {"incident_id": "INC-2894", "date": "2025-07-30", "summary": "Redis resync causing eviction storm"},
            ),
            runbook_hint="Monitor resync progress; consider read traffic throttling or resharding backlog-heavy shard.",
        ),
        IncidentDefinition(
            incident_id=15,
            incident_type="novel_mtls",
            severity="critical",
            service="api-gateway",
            host="ip-10-0-4-45",
            ground_truth="mtls_cert_mismatch",
            logs=logs["novel_mtls"],
            tags=("env:prod", "service:api-gateway", "tls", "region:us-east-1"),
            metadata={"novel": True, "variant": "follow-up"},
            metrics=(
                {"metric": "envoy.tls.handshake_failures", "value": 3, "baseline": 1},
                {"metric": "http.error_rate_pct", "value": 1.8, "baseline": 0.7},
            ),
            recent_changes=(
                {"change_id": "cert_rotation_verify_1130", "author": "noah.stevens", "timestamp": "2025-10-11T15:05:00Z", "summary": "Validated new bundle rollout across payment-router"},
            ),
            related_incidents=(
                {"incident_id": "INC-3301", "date": "2025-09-03", "summary": "mTLS chain mismatch after partial cert rotation"},
                {"incident_id": "INC-3306", "date": "2025-09-05", "summary": "Follow-up confirmation after cert redeploy"},
            ),
            runbook_hint="Confirm promoted persona memory instructions applied: verify cert chain, reload envoy with new bundle.",
        ),
        IncidentDefinition(
            incident_id=16,
            incident_type="network_partition",
            severity="major",
            service="api-gateway",
            host="ip-10-0-7-21",
            ground_truth="transit_link_congestion",
            logs=logs["network_partition"],
            tags=("env:prod", "service:api-gateway", "network"),
            metrics=(
                {"metric": "network.transit_util_pct", "value": 92, "baseline": 55},
            ),
            recent_changes=(
                {"change_id": "traffic_shift_210", "author": "matt.rogers", "timestamp": "2025-10-11T09:55:00Z", "summary": "Shifted 30% traffic to transit link AZ2"},
            ),
            related_incidents=(
                {"incident_id": "INC-2942", "date": "2025-06-25", "summary": "Transit congestion causing Packet loss"},
            ),
            runbook_hint="Throttle traffic on congested link; coordinate with network ops to increase capacity.",
        ),
    ]
    return defn


def render_incident(defn: IncidentDefinition, *, start_at: datetime | None = None) -> dict[str, Any]:
    logs: list[str] = []
    cursor = start_at
    ddtags = _format_ddtags(defn.tags)

    base = start_at or datetime.now(timezone.utc)
    for index, template in enumerate(defn.logs):
        if index == 0:
            cursor = base
        else:
            cursor = cursor + timedelta(seconds=max(template.delay_seconds, 0.5))
        timestamp = cursor.isoformat().replace("+00:00", "Z")
        record = {
            "timestamp": timestamp,
            "service": defn.service,
            "host": defn.host,
            "ddsource": template.source,
            "ddtags": ddtags,
            "status": template.status,
            "message": template.message,
        }
        if template.tags:
            record["tags"] = list(template.tags)
        if template.extra:
            record.update(template.extra)
        logs.append(json.dumps(record, ensure_ascii=False))

    metadata = {
        "incident_type": defn.incident_type,
        "severity": defn.severity,
        "tenant_id": TENANT_ID,
        "service": defn.service,
        "tags": list(defn.tags),
    }
    metadata.update(defn.metadata)

    return {
        "incident_id": defn.incident_id,
        "logs": logs,
        "metadata": metadata,
        "ground_truth": defn.ground_truth,
        "metrics": list(defn.metrics),
        "recent_changes": list(defn.recent_changes),
        "related_incidents": list(defn.related_incidents),
        "runbook_hint": defn.runbook_hint,
    }


def iter_incident_plan() -> Iterator[IncidentDefinition]:
    """Return an iterator over the default incident plan."""

    return iter(_default_incident_plan())


async def generate_incident_stream(
    *,
    playback: str = "slow",
    incident_plan: Iterable[IncidentDefinition] | None = None,
    loop_forever: bool = False,
    interval_override: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield incident payloads at the requested playback cadence."""

    playback = playback.lower()
    cadence = {
        "slow": 2.0,
        "fast": 0.75,
        "replay": 0.1,
    }.get(playback, 2.0)
    if interval_override is not None and interval_override > 0:
        cadence = interval_override

    plan = list(incident_plan or _default_incident_plan())
    total = len(plan)
    iteration = 0

    while True:
        for index, definition in enumerate(plan):
            base_time = datetime.now(timezone.utc)
            incident = render_incident(
                definition,
                start_at=base_time,
            )
            if iteration:
                incident["incident_id"] = definition.incident_id + iteration * total
            yield incident
            await asyncio.sleep(cadence)
        iteration += 1
        if not loop_forever:
            break


async def _stream_to_console(args: argparse.Namespace) -> None:
    console = Console()
    incident_iter = generate_incident_stream(
        playback=args.speed,
        interval_override=args.interval,
        loop_forever=args.loop,
    )
    async for incident in incident_iter:
        console.rule(f"[bold red]Incident #{incident['incident_id']} — {incident['metadata']['incident_type']}")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Service", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Message", style="white")
        for raw in incident["logs"]:
            record = json.loads(raw)
            table.add_row(
                record.get("timestamp", ""),
                record.get("service", ""),
                record.get("status", ""),
                record.get("message", ""),
            )
            await asyncio.sleep(args.log_interval)
        console.print(table)
        if args.once:
            break


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream synthetic SRE incidents to the console.")
    parser.add_argument(
        "--speed",
        choices=("slow", "fast", "replay"),
        default="slow",
        help="Playback profile controlling the delay between incidents.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Override the delay between incidents (seconds).",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=0.4,
        help="Delay between individual log lines when printing (seconds).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay incidents indefinitely.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Emit a single incident and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_stream_to_console(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
