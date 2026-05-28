"""
Fetch stellar data from the Gaia DR3 archive.

The Gaia satellite measures precise positions and brightness of ~1.8 billion stars.
Data is queried using ADQL (Astronomical Data Query Language), a dialect of SQL
used by the Gaia Archive and other VO (Virtual Observatory) services.
"""

import ssl
import warnings
import pandas as pd
import astropy.utils.data as aud
from astroquery.gaia import Gaia


def _configure_network(timeout: int = 60) -> None:
    """
    Apply network settings needed to reach the Gaia archive.

    The SSL override is required in environments with corporate network inspection
    (e.g., man-in-the-middle proxies) that break astroquery's certificate chain.
    Without it, queries fail with SSL verification errors.

    WARNING: this mutates the process-global SSL context — all subsequent HTTPS
    connections in this Python process will skip certificate verification.
    Prefer setting SSL_CERT_FILE or REQUESTS_CA_BUNDLE to your corporate CA bundle
    instead. Only use this override if that is not possible.
    """
    aud.REMOTE_TIMEOUT = timeout
    warnings.warn(
        "SSL certificate verification is disabled for this session. "
        "All HTTPS connections in this process will skip cert checks. "
        "Set SSL_CERT_FILE to your corporate CA bundle to avoid this.",
        UserWarning,
        stacklevel=3,
    )
    ssl._create_default_https_context = ssl._create_unverified_context


def fetch_gaia_sample(top_n: int = 10000) -> pd.DataFrame:
    """
    Query the Gaia DR3 catalog and return a raw DataFrame.

    Columns returned:
        source_id       — Unique Gaia identifier for each star
        ra, dec         — Sky coordinates (degrees)
        parallax        — Angular shift in milliarcseconds; used to compute distance
        parallax_error  — Measurement uncertainty on parallax
        phot_g_mean_mag — Apparent brightness in the G (green) broadband filter
        bp_rp           — Color index: Blue Photometer minus Red Photometer.
                          Low bp_rp = hot blue star, high bp_rp = cool red star.
        pmra            — Proper motion in RA direction (mas/yr), includes cos(dec) factor
        pmdec           — Proper motion in Dec direction (mas/yr)

    Args:
        top_n: Maximum number of stars to return (default 10,000). Must be a positive int.

    Returns:
        pandas.DataFrame with the columns above.

    Raises:
        TypeError: if top_n is not an integer.
        RuntimeError: if the Gaia archive query fails.
    """
    if not isinstance(top_n, int) or top_n <= 0:
        raise TypeError(f"top_n must be a positive integer, got {top_n!r}")

    _configure_network()

    # ADQL query — SELECT TOP limits rows; WHERE filters out stars with no parallax
    # (parallax is essential for computing distance, so rows without it are unusable)
    query = f"""
    SELECT TOP {top_n}
        source_id,
        ra,
        dec,
        parallax,
        parallax_error,
        phot_g_mean_mag,
        bp_rp,
        pmra,
        pmdec
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
    """

    try:
        job = Gaia.launch_job(query)
        return job.get_results().to_pandas()
    except Exception as exc:
        raise RuntimeError(
            f"Gaia archive query failed. Check your network/VPN connection. Original error: {exc}"
        ) from exc
