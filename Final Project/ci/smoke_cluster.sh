#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-giftcard-sentiment}"

kubectl -n "$NAMESPACE" rollout status deploy/giftcard-api --timeout=120s
kubectl -n "$NAMESPACE" rollout status deploy/giftcard-ui --timeout=120s

API_POD="$(kubectl -n "$NAMESPACE" get pod -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NAMESPACE" exec "$API_POD" -- python /app/ci/e2e_api.py
