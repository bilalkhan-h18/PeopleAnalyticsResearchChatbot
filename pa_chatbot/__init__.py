"""People analytics research assistant."""

# Use the operating system's certificate store (e.g. Windows) for HTTPS.
# Corporate networks often re-sign traffic with their own root certificate,
# which Windows trusts but Python's bundled list doesn't, causing
# CERTIFICATE_VERIFY_FAILED. This runs before any connection is made.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
