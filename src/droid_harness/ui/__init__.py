"""droid-harness Studio: the live annotated view (``--ui`` / ``watch``) and the replay inspector (``inspect``).

One zero-build browser front end (``static/studio.html``) served by the standard library's HTTP
server, so it works headless and over SSH (``ssh -L 8765:localhost:8765 veritas``). ``--native``
opens the same page in a desktop window when ``pywebview`` is installed (``pip install -e ".[gui]"``).

Live and replay produce the same snapshot schema (``snapshot.py``), so the page has one renderer.
The brain thread only hands references to a lock-protected slot (``telemetry.TelemetryTap``); JSON
building and JPEG encoding happen on a separate publisher thread at a capped rate.
"""
