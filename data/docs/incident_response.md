# Incident Response

Severities at Akamai:
- SEV1: total outage or data loss affecting all customers. Page on-call immediately.
- SEV2: major feature broken or degradation affecting a significant fraction of users.
- SEV3: minor bug or single-customer issue. File a ticket; no page.

To declare an incident, type `/incident declare` in Slack. This creates a dedicated channel, pages the on-call, and starts a status page draft. The first responder becomes Incident Commander until handed off.

Communication cadence: SEV1 updates every 15 minutes on the status page and in #incidents. SEV2 updates every 30 minutes. All updates must include impact, current actions, and ETA if known.

Postmortems are required for all SEV1 and SEV2 incidents and must be published within 5 business days. Use the postmortem template in Notion. Postmortems are blameless and focus on systemic causes.
