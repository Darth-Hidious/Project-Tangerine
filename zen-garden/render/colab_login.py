"""Sign the Google Colab CLI in from a headless machine, in two steps that can be hours apart.

    python render/colab_login.py url            # prints the Google sign-in URL
    python render/colab_login.py code <CODE>    # exchanges the code Google shows after sign-in

It runs the same flow as `colab --auth=oauth2` (the CLI's own OAuth client, its registered
landing page that displays the code, PKCE), but keeps the verifier in a file between the two
steps instead of waiting on a prompt, so an agent can hand you the URL and take the code later.
The token lands where the CLI looks for it (~/.config/colab-cli/token.json); after that every
`colab --auth=oauth2 ...` command runs without asking again.

Scopes requested (the CLI's): your Google profile and email, Colab, Google Cloud, and Drive
files the CLI itself creates. Needs the CLI installed: pip install google-colab-cli (Python 3.12+).
"""

from __future__ import annotations

import json
import os
import sys
from importlib import resources

from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI, TOKEN_CONFIG_PATH
from google_auth_oauthlib.flow import InstalledAppFlow

PENDING = os.path.join(os.path.dirname(TOKEN_CONFIG_PATH), "pending_login.json")


def _flow(verifier: str | None = None) -> InstalledAppFlow:
    config = json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())
    flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES, code_verifier=verifier,
                                               autogenerate_code_verifier=verifier is None)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    return flow


def main(argv: list[str]) -> int:
    if argv[:1] == ["url"]:
        flow = _flow()
        url, state = flow.authorization_url(prompt="consent", token_usage="remote")
        os.makedirs(os.path.dirname(PENDING), exist_ok=True)
        with open(PENDING, "w") as f:
            json.dump({"verifier": flow.code_verifier, "state": state}, f)
        os.chmod(PENDING, 0o600)
        print(url)
        return 0
    if argv[:1] == ["code"] and len(argv) == 2:
        with open(PENDING) as f:
            pending = json.load(f)
        flow = _flow(pending["verifier"])
        flow.fetch_token(code=argv[1].strip())
        with open(TOKEN_CONFIG_PATH, "w") as f:
            f.write(flow.credentials.to_json())
        os.chmod(TOKEN_CONFIG_PATH, 0o600)
        os.remove(PENDING)
        print("signed in; token saved to", TOKEN_CONFIG_PATH)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
