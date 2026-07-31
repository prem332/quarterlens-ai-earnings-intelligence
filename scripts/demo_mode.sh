#!/usr/bin/env bash
# Toggle the production Container App between cost mode and demo mode.
#
#   ./scripts/demo_mode.sh on    # warm standby — no cold start, bills continuously
#   ./scripts/demo_mode.sh off   # scale-to-zero — free when idle, cold start on wake
#   ./scripts/demo_mode.sh status
#
# Why this exists
# ---------------
# min-replicas=0 (the default, "off") is already the cheap setting: the app
# sleeps after ~5 minutes idle and bills nothing while asleep. The cost is that
# waking it pays image pull + torch/transformers imports + cross-encoder and
# FinBERT loading -- measured at 17s minimum, 46s under load -- before the
# request is even served.
#
# min-replicas=1 ("on") removes that entirely, but bills the whole time. At
# 2 vCPU / 4 GiB that is ~172,800 vCPU-seconds/day, which exhausts the monthly
# Container Apps free grant in roughly one day. Do NOT leave it on.
#
# Intended use: flip on a few minutes before a demo or interview, flip off
# straight after.
set -euo pipefail

APP=quarterlens-api
RG=quarterlens-phase1-rg

case "${1:-status}" in
  on)
    echo "Demo mode ON — warm standby, no cold start (BILLS CONTINUOUSLY)"
    az containerapp update -n "$APP" -g "$RG" --min-replicas 1 -o none
    URL=$(az containerapp show -n "$APP" -g "$RG" \
          --query "properties.configuration.ingress.fqdn" -o tsv)
    echo "Warming up; first request may still be slow until the replica is ready..."
    curl -s --retry 30 --retry-delay 5 --retry-connrefused \
         -o /dev/null -w "health: HTTP %{http_code}\n" "https://$URL/api/health"
    echo "Ready: https://$URL"
    echo "REMEMBER: ./scripts/demo_mode.sh off  when you are done."
    ;;
  off)
    echo "Demo mode OFF — scale-to-zero, free when idle"
    az containerapp update -n "$APP" -g "$RG" --min-replicas 0 -o none
    echo "Done. The app sleeps after ~5 minutes idle."
    ;;
  status)
    az containerapp show -n "$APP" -g "$RG" --query \
      "{minReplicas:properties.template.scale.minReplicas,
        maxReplicas:properties.template.scale.maxReplicas,
        cpu:properties.template.containers[0].resources.cpu,
        memory:properties.template.containers[0].resources.memory,
        url:properties.configuration.ingress.fqdn}" -o yaml
    ;;
  *)
    echo "usage: $0 {on|off|status}" >&2
    exit 1
    ;;
esac
