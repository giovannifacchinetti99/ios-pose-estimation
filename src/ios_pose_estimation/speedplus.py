"""
Access to the SPEED+ dataset.

SPEED+ (Park, Märtens, Lecuyer, Izzo, D'Amico, 2021) is the dataset of the
2021 Satellite Pose Estimation Competition of ESA's Kelvins platform. It
contains images of the Tango spacecraft of the PRISMA mission, each labeled
with the pose of the spacecraft with respect to the camera: 59,960
synthetic images, split into training and validation, and two sets taken
in a laboratory with a mock-up under realistic illumination, lightbox and
sunlamp.

The dataset is distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 license, as stated in the
LICENSE.md file inside the archive. It can be used here to learn and to
validate methods; it cannot be the basis of a commercial product, and its
images are not redistributed with this package.

The archive is a single file of 17 GB. Reading it does not require
downloading it: the server accepts requests for a range of bytes, and a
zip archive keeps its index at the end, so the index and then any single
file can be fetched on their own. SpeedPlus keeps a local folder with the
files already fetched and fetches the missing ones on demand.
"""

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ios_pose_estimation.camera import PinholeCamera

ARCHIVE_URL = "https://zenodo.org/api/records/5588480/files/speedplusv2.zip/content"
ARCHIVE_ROOT = "speedplusv2"

# The 11 reference points of the Tango spacecraft used by the baseline
# studies of SPEED+, in the body frame [m]: the four corners of the top
# panel, four points of the bottom, and the tips of three antennas.
# Source: tangoPoints.mat in github.com/tpark94/speedplusbaseline (MIT
# license, SLAB Group, Stanford University).
TANGO_KEYPOINTS = np.array([
    [-0.3700, -0.3850, 0.3215],
    [-0.3700, 0.3850, 0.3215],
    [0.3700, 0.3850, 0.3215],
    [0.3700, -0.3850, 0.3215],
    [-0.3700, -0.2640, 0.0000],
    [-0.3700, 0.3040, 0.0000],
    [0.3700, 0.3040, 0.0000],
    [0.3700, -0.2640, 0.0000],
    [-0.5427, 0.4877, 0.2535],
    [0.5427, 0.4877, 0.2591],
    [0.3050, -0.5790, 0.2515],
])

# pairs of keypoints joined by an edge of the body, for drawing
TANGO_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

DOMAINS = {
    ("synthetic", "train"): "synthetic/train.json",
    ("synthetic", "validation"): "synthetic/validation.json",
    ("lightbox", "test"): "lightbox/test.json",
    ("sunlamp", "test"): "sunlamp/test.json",
}


def http_range(url):
    """A function that fetches a range of bytes of a file over HTTP."""

    def fetch(start, end):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    return fetch


def http_size(url):
    """Size of a remote file [bytes], from the reply to a request for its first byte."""

    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Range"].split("/")[-1])


class RangeReader(io.RawIOBase):
    """
    A read-only file made of byte ranges fetched on demand.

    It gives the standard file interface (seek, tell, read) to a file
    that is somewhere else and can only be asked for pieces, which is
    enough for the zipfile module to open a remote archive: the module
    reads the index at the end of the file and then only the bytes of
    the members that are extracted. Each fetch asks for at least a block,
    so that the many small reads that follow one another while a member
    is decompressed are served from memory.

    Parameters
    ----------
    fetch : callable
        Function (start, end) returning the bytes from start to end,
        both included.
    size : int
        Size of the file [bytes].
    block : int
        Smallest number of bytes asked for in one fetch.

    Author: Giovanni Facchinetti, 2026
    """

    def __init__(self, fetch, size, block=1 << 20):
        super().__init__()
        self._fetch = fetch
        self._size = size
        self._block = block
        self._position = 0
        self._cache_start = 0
        self._cache = b""
        self.fetches = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        else:
            self._position = self._size + offset
        return self._position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        if size <= 0:
            return b""

        end = self._position + size
        cached = self._cache_start <= self._position and end <= self._cache_start + len(self._cache)
        if not cached:
            last = min(self._size, max(end, self._position + self._block)) - 1
            self._cache = self._fetch(self._position, last)
            self._cache_start = self._position
            self.fetches += 1

        start = self._position - self._cache_start
        self._position = end
        return self._cache[start:start + size]


def extract_member(blob, offset, info):
    """
    The content of a member of a zip archive, from bytes of the archive.

    Parameters
    ----------
    blob : bytes
        Bytes of the archive that contain the member.
    offset : int
        Position in `blob` of the local header of the member.
    info : zipfile.ZipInfo
        Entry of the member in the index of the archive.
    """

    import zlib

    header = blob[offset:offset + 30]
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"no local header for {info.filename}")
    name_length = int.from_bytes(header[26:28], "little")
    extra_length = int.from_bytes(header[28:30], "little")
    start = offset + 30 + name_length + extra_length
    data = blob[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"corrupted member {info.filename}")
    return data


@dataclass
class PoseLabel:
    """
    The label of one image.

    Attributes
    ----------
    filename : str
        Name of the image file.
    quaternion : numpy.ndarray, shape (4,) or None
        Rotation from the camera frame to the body frame of the target,
        scalar part first.
    translation : numpy.ndarray, shape (3,) or None
        Position of the body origin of the target in the camera frame
        [m].
    """

    filename: str
    quaternion: np.ndarray = None
    translation: np.ndarray = None

    @property
    def distance(self):
        return float(np.linalg.norm(self.translation))


class SpeedPlus:
    """
    The SPEED+ dataset, read from a local folder that is filled on demand
    from the remote archive.

    The folder has the layout of the archive: camera.json, and for each
    domain a file of labels and a folder of images. A file that is not
    there is fetched from the archive the first time it is asked for,
    unless remote access is disabled. A complete local copy of the
    dataset, unpacked in the folder, works without any remote access.

    Parameters
    ----------
    root : str or pathlib.Path
        Local folder.
    remote : bool
        Whether missing files may be fetched from the archive.
    url : str
        Address of the archive.

    Author: Giovanni Facchinetti, 2026
    Reference: Park, Märtens, Lecuyer, Izzo, D'Amico, "SPEED+: Next-
    Generation Dataset for Spacecraft Pose Estimation across Domain Gap,"
    IEEE Aerospace Conference, 2022
    """

    def __init__(self, root, remote=True, url=ARCHIVE_URL):
        self.root = Path(root)
        self.remote = remote
        self.url = url
        self._archive = None

    def _open_archive(self):
        if self._archive is None:
            self._fetch = http_range(self.url)
            reader = RangeReader(self._fetch, http_size(self.url))
            self._archive = zipfile.ZipFile(reader)
        return self._archive

    def archive_images(self, domain):
        """
        The images of a domain in the order in which they are stored in the
        archive, which is not the order of their names.

        Returns
        -------
        list of zipfile.ZipInfo
        """

        prefix = f"{ARCHIVE_ROOT}/{domain}/images/"
        members = [info for info in self._open_archive().infolist()
                   if info.filename.startswith(prefix) and not info.is_dir()]
        return sorted(members, key=lambda info: info.header_offset)

    def prefetch(self, domain, count, start=0, block=64 << 20, workers=4):
        """
        Fetch many images of a domain with a few large requests.

        The images of the archive are stored one after the other with no
        gap, so a run of consecutive members is a single range of bytes.
        The run of `count` images starting at position `start` in the
        order of the archive is fetched in blocks of about `block` bytes,
        `workers` blocks at a time, and every image not yet in the local
        folder is written there. Since the order of the archive is
        unrelated to the poses, a run is a random sample of the domain.

        Returns
        -------
        list of str
            Names of the images of the run.

        Author: Giovanni Facchinetti, 2026
        """

        from concurrent.futures import ThreadPoolExecutor

        members = self.archive_images(domain)[start:start + count]
        groups, current, size = [], [], 0
        for info in members:
            current.append(info)
            size += info.compress_size + 30 + len(info.filename) + 64
            if size >= block:
                groups.append(current)
                current, size = [], 0
        if current:
            groups.append(current)

        def fetch_group(group):
            missing = [info for info in group
                       if not (self.root / info.filename[len(ARCHIVE_ROOT) + 1:]).exists()]
            if not missing:
                return
            first = missing[0].header_offset
            last = missing[-1].header_offset + missing[-1].compress_size + 30 + len(missing[-1].filename) + 1024
            blob = self._fetch(first, last)
            for info in missing:
                data = extract_member(blob, info.header_offset - first, info)
                local = self.root / info.filename[len(ARCHIVE_ROOT) + 1:]
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(fetch_group, groups))
        return [info.filename.rsplit("/", 1)[-1] for info in members]

    def path(self, member):
        """
        Local path of a file of the dataset, fetched if it is missing.

        Parameters
        ----------
        member : str
            Path inside the dataset, for example "camera.json" or
            "synthetic/images/img000001.jpg".
        """

        local = self.root / member
        if local.exists():
            return local
        if not self.remote:
            raise FileNotFoundError(f"{local} is missing and remote access is disabled")

        data = self._open_archive().read(f"{ARCHIVE_ROOT}/{member}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return local

    def camera(self):
        """The camera of the dataset, with its calibrated distortion."""

        with open(self.path("camera.json"), encoding="utf-8") as stream:
            data = json.load(stream)

        matrix = np.array(data["cameraMatrix"], dtype=float)
        return PinholeCamera(
            fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2],
            width=int(data["Nu"]), height=int(data["Nv"]),
            distortion=np.array(data["distCoeffs"], dtype=float),
        )

    def labels(self, domain="synthetic", split="train"):
        """
        Labels of a domain.

        Parameters
        ----------
        domain : str
            "synthetic", "lightbox" or "sunlamp".
        split : str
            "train" or "validation" for the synthetic domain, "test" for
            the other two.

        Returns
        -------
        list of PoseLabel
            The pose is None for images distributed without a label.
        """

        with open(self.path(DOMAINS[(domain, split)]), encoding="utf-8") as stream:
            entries = json.load(stream)

        labels = []
        for entry in entries:
            quaternion = entry.get("q_vbs2tango_true")
            translation = entry.get("r_Vo2To_vbs_true")
            labels.append(PoseLabel(
                filename=entry["filename"],
                quaternion=None if quaternion is None else np.array(quaternion, dtype=float),
                translation=None if translation is None else np.array(translation, dtype=float),
            ))
        return labels

    def image(self, domain, filename):
        """
        An image as an array of gray levels.

        Returns
        -------
        numpy.ndarray, shape (height, width), dtype uint8
        """

        from PIL import Image

        with Image.open(self.path(f"{domain}/images/{filename}")) as picture:
            return np.asarray(picture.convert("L"))
