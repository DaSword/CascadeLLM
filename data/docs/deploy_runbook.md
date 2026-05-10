# Deploy Runbook

Akamai uses GitOps via ArgoCD for all production deploys. Merging to the main branch triggers an automatic deploy to staging. Production deploys are gated by a manual approval in ArgoCD.

Production deploy windows are Monday through Thursday, 10 AM to 4 PM Pacific. No production deploys on Fridays, weekends, or company holidays without an approved exception in #release-approvals.

Pre-deploy checklist: tests passing in CI, staging smoke tests green for at least 2 hours, runbook updated if behavior changes, on-call notified in #oncall, and feature flags configured if rolling out gradually.

If a deploy needs to be rolled back, click "Rollback" in ArgoCD on the affected service. The rollback is automatic and takes 60-90 seconds. File a postmortem within 48 hours of any rollback.
