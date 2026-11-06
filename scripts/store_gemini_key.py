"""Helper for setting up the two secrets this agent needs, without ever
pasting a raw key into a shared shell history if you use the
Credential-Manager option.

Two independent things this script can do:

  1. Generate a new random VISION_AGENT_API_KEY (this is the key ERA/Bridge
     use to call this agent -- it is NOT the Gemini key).
  2. Store your GEMINI_API_KEY in Windows Credential Manager (via the
     optional `keyring` package) instead of a plaintext .env file, and
     print the two lines to put in your .env so main.py picks it up.

Usage:

    python scripts\\store_gemini_key.py --gen-vision-key
    python scripts\\store_gemini_key.py --store-gemini-key
    python scripts\\store_gemini_key.py --help
"""

from __future__ import annotations

import argparse
import getpass
import sys

sys.path.insert(0, ".")

from config import generate_api_key  # noqa: E402

SERVICE_NAME = "era_vision_agent"


def gen_vision_key() -> None:
    key = generate_api_key()
    print("\nGenerated VISION_AGENT_API_KEY:\n")
    print(f"  {key}\n")
    print("Add this to your .env file as:\n")
    print(f"  VISION_AGENT_API_KEY={key}\n")
    print(
        "Use this SAME value as the 'X-Vision-Agent-Key' header when ERA/Bridge "
        "calls this agent. It is unrelated to your Gemini API key."
    )


def store_gemini_key() -> None:
    try:
        import keyring
    except ImportError:
        print(
            "The optional 'keyring' package is not installed. Install it with:\n"
            "  pip install keyring\n"
            "and re-run this command. Alternatively, put GEMINI_API_KEY directly "
            "in your .env file (less secure, but works without extra dependencies).",
            file=sys.stderr,
        )
        sys.exit(1)

    key = getpass.getpass("Paste your Gemini API key (input hidden): ").strip()
    if not key:
        print("No key entered; aborting.", file=sys.stderr)
        sys.exit(1)

    keyring.set_password(SERVICE_NAME, "GEMINI_API_KEY", key)
    print(
        "\nStored in Windows Credential Manager under service "
        f"'{SERVICE_NAME}', account 'GEMINI_API_KEY'.\n"
        "To have config.py read it from there automatically, add this to your "
        ".env instead of the raw key:\n\n"
        "  GEMINI_API_KEY_FROM_KEYRING=1\n\n"
        "(Note: automatic keyring lookup is a small addition to config.py you "
        "can wire in later; by default config.py reads GEMINI_API_KEY from the "
        "environment/.env directly.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen-vision-key", action="store_true", help="Generate a new VISION_AGENT_API_KEY.")
    parser.add_argument("--store-gemini-key", action="store_true", help="Store GEMINI_API_KEY in Windows Credential Manager (requires 'keyring').")
    args = parser.parse_args()

    if not args.gen_vision_key and not args.store_gemini_key:
        parser.print_help()
        return

    if args.gen_vision_key:
        gen_vision_key()
    if args.store_gemini_key:
        store_gemini_key()


if __name__ == "__main__":
    main()
