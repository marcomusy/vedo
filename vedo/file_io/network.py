from __future__ import annotations
"""Network and filesystem utilities for file I/O."""

import os
import time
from tempfile import NamedTemporaryFile

from vedo import colors, settings

__docformat__ = "google"

def download(url: str, force=False, verbose=True) -> str:
    """
    Retrieve a file from a URL, save it locally and return its path.
    Use `force=True` to force a reload and discard cached copies.
    """
    if not url.startswith("https://"):
        # assume it's a file so no need to download
        return url
    url = url.replace("www.dropbox", "dl.dropbox")

    if "github.com" in url:
        url = url.replace("/blob/", "/raw/")

    basename = os.path.basename(url)

    if "?" in basename:
        basename = basename.split("?")[0]

    home_directory = os.path.expanduser("~")
    cachedir = os.path.join(home_directory, settings.cache_directory, "vedo")
    fname = os.path.join(cachedir, basename)
    # Create the directory if it does not exist
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    if not force and os.path.exists(fname):
        if verbose:
            colors.printc("reusing cached file:", fname)
        return fname

    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if verbose:
            colors.printc("reading", basename, "from", url.split("/")[2][:40], "...", end="")

    except ImportError:
        import urllib2 # type: ignore
        import contextlib
        urlopen = lambda url_: contextlib.closing(urllib2.urlopen(url_))
        req = url
        if verbose:
            colors.printc("reading", basename, "from", url.split("/")[2][:40], "...", end="")

    with urlopen(req) as response, open(fname, "wb") as output:
        output.write(response.read())

    if verbose:
        colors.printc(" done.")
    return fname


########################################################################
# def download_new(url, to_local_file="", force=False, verbose=True):
#     """
#     Downloads a file from `url` to `to_local_file` if the local copy is outdated.

#     Arguments:
#         url : (str)
#             The URL to download the file from.
#         to_local_file : (str)
#             The local file name to save the file to.
#             If not specified, the file name will be the same as the remote file name
#             in the directory specified by `settings.cache_directory + "/vedo"`.
#         force : (bool)
#             Force a new download even if the local file is up to date.
#         verbose : (bool)
#             Print verbose messages.
#     """
#     if not url.startswith("https://"):
#         if os.path.exists(url):
#             # Assume the url is already the local file path
#             return url
#         else:
#             raise FileNotFoundError(f"File not found: {url}")

#     from datetime import datetime
#     import requests

#     url = url.replace("www.dropbox", "dl.dropbox")

#     if "github.com" in url:
#         url = url.replace("/blob/", "/raw/")

#     # Get the user's home directory
#     home_directory = os.path.expanduser("~")

#     # Define the path for the cache directory
#     cachedir = os.path.join(home_directory, settings.cache_directory, "vedo")

#     # Create the directory if it does not exist
#     if not os.path.exists(cachedir):
#         os.makedirs(cachedir)

#     if not to_local_file:
#         basename = os.path.basename(url)
#         if "?" in basename:
#             basename = basename.split("?")[0]
#         to_local_file = os.path.join(cachedir, basename)
#         if verbose: print(f"Using local file name: {to_local_file}")

#     # Check if the local file exists and get its last modified time
#     if os.path.exists(to_local_file):
#         to_local_file_modified_time = os.path.getmtime(to_local_file)
#     else:
#         to_local_file_modified_time = 0

#     # Send a HEAD request to get last modified time of the remote file
#     response = requests.head(url)
#     if 'Last-Modified' in response.headers:
#         remote_file_modified_time = datetime.strptime(
#             response.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S GMT'
#         ).timestamp()
#     else:
#         # If the Last-Modified header not available, assume file needs to be downloaded
#         remote_file_modified_time = float('inf')

#     # Download the file if the remote file is newer
#     if force or remote_file_modified_time > to_local_file_modified_time:
#         response = requests.get(url)
#         with open(to_local_file, 'wb') as file:
#             file.write(response.content)
#             if verbose: print(f"Downloaded file from {url} -> {to_local_file}")
#     else:
#         if verbose: print("Local file is up to date.")
#     return to_local_file


########################################################################
def gunzip(filename: str) -> str:
    """Unzip a `.gz` file to a temporary file and returns its path."""
    if not filename.endswith(".gz"):
        # colors.printc("gunzip() error: file must end with .gz", c='r')
        return filename

    import gzip

    tmp_file = NamedTemporaryFile(delete=False)
    tmp_file.name = os.path.join(
        os.path.dirname(tmp_file.name), os.path.basename(filename).replace(".gz", "")
    )
    inF = gzip.open(filename, "rb")
    with open(tmp_file.name, "wb") as outF:
        outF.write(inF.read())
    inF.close()
    return tmp_file.name

########################################################################
def file_info(file_path: str) -> tuple[str, str]:
    """Return the file size and creation time of input file"""
    siz, created = "", ""
    if os.path.isfile(file_path):
        f_info = os.stat(file_path)
        num = f_info.st_size
        for x in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024.0:
                break
            num /= 1024.0
        siz = "%3.1f%s" % (num, x)
        created = time.ctime(os.path.getmtime(file_path))
    return siz, created

__all__ = ["download", "gunzip", "file_info"]
