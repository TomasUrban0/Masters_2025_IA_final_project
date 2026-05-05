# AutoTor

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)

## About

High-level Python API for automated anonymous web requests through Tor. Supports parallel threaded requests with automatic circuit renewal and randomised user-agents.

### Features

- Automated Tor download and initialization (Windows)
- Multi-process Tor circuit management
- Thread pool-based parallel requests with configurable workers
- Automatic IP rotation per request cycle
- Randomised user-agent headers
- Context manager for automatic resource cleanup

### Dependencies

- [requests\[socks\]](https://docs.python-requests.org/) (>=2.27.1)
- [stem](https://stem.torproject.org/) (>=1.8.0)
- [fake-useragent](https://github.com/hellysmile/fake-useragent) (>=0.1.11)

## Installation

```bash
# From source
pip install .

# Or from PyPI
pip install autotor
```

## Usage

### 1. Create a derived class

```python
from autotor import TorRequests
from threading import Lock

LOCK = Lock()

class MyScraper(TorRequests):

    def __init__(self, target_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_url = target_url
        self.results = []

    def request(self, elem, n_id):
        session = self.get_tor_session(n_id)
        self.renew_tor_ip(n_id)
        response = session.get(f"{self.target_url}/{elem}")
        with LOCK:
            self.results.append(response.text)
```

### 2. Run parallel requests

```python
with MyScraper(target_url="https://example.com", n_process=5, tor_root=".") as tor:
    tor.threaded_request(range(100))
    print(f"Collected {len(tor.results)} responses")
```

### Module Structure

| File | Description |
|------|-------------|
| `src/autotor/autotor_base.py` | Core `TorRequests` class — Tor lifecycle, sessions, thread pool |
| `src/autotor/autotor_ip.py` | Example implementation: IP verification through Tor |
| `src/main.py` | Usage example |

## License

MIT — see [LICENSE](LICENSE) for details.

**Disclaimer:** The authors are not responsible for misuse of this API. Use responsibly and in compliance with applicable laws.
