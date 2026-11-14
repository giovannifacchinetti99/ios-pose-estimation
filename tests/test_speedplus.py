import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from ios_pose_estimation import speedplus
from ios_pose_estimation.speedplus import (
    ARCHIVE_ROOT,
    TANGO_EDGES,
    TANGO_KEYPOINTS,
    RangeReader,
    SpeedPlus,
)

CAMERA = {
    "Nu": 1920, "Nv": 1200, "ppx": 5.86e-6, "ppy": 5.86e-6, "fx": 0.0175, "fy": 0.0175, "ccx": 960, "ccy": 600,
    "cameraMatrix": [[2988.5, 0, 960], [0, 2988.3, 600], [0, 0, 1]],
    "distCoeffs": [-0.22, 0.51, -0.0007, -0.0002, -0.13],
}
TRAIN = [
    {"filename": "img000001.jpg", "q_vbs2tango_true": [-0.336458, -0.093409, -0.8286, 0.437599],
     "r_Vo2To_vbs_true": [-0.129896, 0.069519, 6.457073]},
    {"filename": "img000002.jpg", "q_vbs2tango_true": [1.0, 0.0, 0.0, 0.0], "r_Vo2To_vbs_true": [0.0, 0.0, 3.0]},
]


def _picture_bytes():
    pixels = (np.arange(12 * 20, dtype=np.uint8).reshape(12, 20) * 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue(), pixels


def _archive_bytes():
    picture, _ = _picture_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{ARCHIVE_ROOT}/camera.json", json.dumps(CAMERA))
        archive.writestr(f"{ARCHIVE_ROOT}/synthetic/train.json", json.dumps(TRAIN))
        archive.writestr(f"{ARCHIVE_ROOT}/lightbox/test.json", json.dumps([{"filename": "img000001.jpg"}]))
        archive.writestr(f"{ARCHIVE_ROOT}/synthetic/images/img000001.jpg", picture)
        archive.writestr(f"{ARCHIVE_ROOT}/padding.bin", bytes(200000))
    return buffer.getvalue()


def _reader(data, block=1 << 20):
    return RangeReader(lambda start, end: data[start:end + 1], len(data), block=block)


def test_range_reader_behaves_like_a_file():
    data = bytes(range(256)) * 40
    reader = _reader(data, block=64)

    assert reader.read(10) == data[:10]
    assert reader.tell() == 10
    reader.seek(-5, io.SEEK_END)
    assert reader.read() == data[-5:]
    assert reader.read(10) == b""
    reader.seek(100)
    reader.seek(20, io.SEEK_CUR)
    assert reader.read(3) == data[120:123]


def test_range_reader_serves_consecutive_small_reads_from_one_fetch():
    data = bytes(range(256)) * 400
    reader = _reader(data, block=4096)

    pieces = [reader.read(16) for _ in range(200)]

    assert b"".join(pieces) == data[:3200]
    assert reader.fetches == 1


def test_a_zip_archive_can_be_read_through_ranges():
    data = _archive_bytes()
    reader = _reader(data, block=2048)

    archive = zipfile.ZipFile(reader)

    assert f"{ARCHIVE_ROOT}/camera.json" in archive.namelist()
    assert json.loads(archive.read(f"{ARCHIVE_ROOT}/camera.json")) == CAMERA
    # the index and one small member, not the whole archive
    assert reader.fetches <= 6


@pytest.fixture
def remote_dataset(tmp_path, monkeypatch):
    data = _archive_bytes()
    monkeypatch.setattr(speedplus, "http_size", lambda url: len(data))
    monkeypatch.setattr(speedplus, "http_range", lambda url: (lambda start, end: data[start:end + 1]))
    return SpeedPlus(tmp_path / "speedplus")


def test_missing_files_are_fetched_once_and_kept(remote_dataset):
    local = remote_dataset.root / "camera.json"
    assert not local.exists()

    camera = remote_dataset.camera()

    assert local.exists()
    assert camera.fx == 2988.5 and camera.fy == 2988.3 and camera.cx == 960 and camera.cy == 600
    assert (camera.width, camera.height) == (1920, 1200)
    assert np.allclose(camera.distortion, CAMERA["distCoeffs"])

    # a second dataset on the same folder needs no remote access at all
    offline = SpeedPlus(remote_dataset.root, remote=False)
    assert offline.camera().fx == 2988.5


def test_labels_with_and_without_a_pose(remote_dataset):
    train = remote_dataset.labels("synthetic", "train")
    test = remote_dataset.labels("lightbox", "test")

    assert [label.filename for label in train] == ["img000001.jpg", "img000002.jpg"]
    assert np.allclose(train[0].quaternion, TRAIN[0]["q_vbs2tango_true"])
    assert np.isclose(train[0].distance, np.linalg.norm(TRAIN[0]["r_Vo2To_vbs_true"]))
    assert train[1].distance == 3.0
    assert test[0].quaternion is None and test[0].translation is None


def test_image_is_returned_as_gray_levels(remote_dataset):
    _, pixels = _picture_bytes()

    image = remote_dataset.image("synthetic", "img000001.jpg")

    assert image.dtype == np.uint8 and image.shape == (12, 20)
    assert np.array_equal(image, pixels)


def test_without_remote_access_a_missing_file_is_an_error(tmp_path):
    dataset = SpeedPlus(tmp_path, remote=False)

    with pytest.raises(FileNotFoundError):
        dataset.camera()


def test_tango_keypoints_describe_a_body_of_the_right_size():
    assert TANGO_KEYPOINTS.shape == (11, 3)

    body = TANGO_KEYPOINTS[:8]
    size = body.max(axis=0) - body.min(axis=0)
    assert np.allclose(size, [0.74, 0.77, 0.3215])
    assert np.allclose(body[:4, 2], 0.3215) and np.allclose(body[4:, 2], 0.0)
    # the antennas stick out of the body
    assert np.all(np.abs(TANGO_KEYPOINTS[8:, :2]).max(axis=1) > 0.48)
    assert all(0 <= a < 8 and 0 <= b < 8 for a, b in TANGO_EDGES) and len(TANGO_EDGES) == 12
