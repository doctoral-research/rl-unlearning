#!/bin/bash
set -e

# If HOST_UID/HOST_GID are set, create a matching user and run as them.
# This ensures files created in mounted volumes have correct ownership.

if [ -n "${HOST_UID}" ] && [ "${HOST_UID}" != "0" ]; then
    # Create group if it doesn't exist
    if ! getent group "${HOST_GID}" > /dev/null 2>&1; then
        groupadd -g "${HOST_GID}" appgroup
    fi
    GROUP_NAME=$(getent group "${HOST_GID}" | cut -d: -f1)

    # Create user if it doesn't exist
    if ! getent passwd "${HOST_UID}" > /dev/null 2>&1; then
        useradd -u "${HOST_UID}" -g "${HOST_GID}" -m -s /bin/bash appuser
    fi
    USER_NAME=$(getent passwd "${HOST_UID}" | cut -d: -f1)

    # Ensure HOME exists and user owns it
    mkdir -p "${HOME:-/tmp/home}"
    chown "${HOST_UID}:${HOST_GID}" "${HOME:-/tmp/home}"
    mkdir -p /workspace/experiments


    exec gosu "${USER_NAME}" "$@"
else
    exec "$@"
fi
