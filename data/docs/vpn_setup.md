# VPN Setup

Akamai uses Tailscale for remote access to internal services. To set up:
1. Download Tailscale from tailscale.com or the Mac App Store / Play Store.
2. Sign in with your Akamai Google account via SSO.
3. Approve the device in your phone's authenticator app.

Once connected, internal hostnames resolve via MagicDNS. You can reach staging at staging.cat.internal and the wiki at wiki.cat.internal without any further configuration.

If you cannot connect, first restart Tailscale. If that fails, check that your device is approved at admin.tailscale.com (IT may need to approve new devices). For persistent issues, file a ticket with #it-helpdesk.

Tailscale must be running for any access to internal tools, including ArgoCD, Grafana, and the wiki.
