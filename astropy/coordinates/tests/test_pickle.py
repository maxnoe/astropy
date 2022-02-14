import pickle

import numpy as np
import pytest

import astropy.units as u
from astropy import coordinates as coord
from astropy.coordinates import (
    ICRS,
    Angle,
    Distance,
    DynamicMatrixTransform,
    Latitude,
    Longitude,
    StaticMatrixTransform,
)
from astropy.tests.helper import check_pickling_recovery, pickle_protocol  # noqa: F401


def test_basic():
    lon1 = Longitude(1.23, "radian", wrap_angle="180d")
    s = pickle.dumps(lon1)
    lon2 = pickle.loads(s)


def test_pickle_longitude_wrap_angle():
    a = Longitude(1.23, "radian", wrap_angle="180d")
    s = pickle.dumps(a)
    b = pickle.loads(s)

    assert a.rad == b.rad
    assert a.wrap_angle == b.wrap_angle


def _dummy_transform(fromcoord, toframe):
    return np.identity(3)


@pytest.mark.parametrize(
    "original",
    [
        Angle(0.0 * u.rad),
        Distance(5 * u.pc),
        DynamicMatrixTransform(_dummy_transform, ICRS, ICRS),
        ICRS(0 * u.rad, 0 * u.rad),
        Latitude(0 * u.rad),
        Longitude(0 * u.rad),
        StaticMatrixTransform(np.identity(3), ICRS, ICRS),
    ],
    ids=lambda x: type(x).__name__,
)
def test_simple_object(pickle_protocol, original):  # noqa: F811
    check_pickling_recovery(original, pickle_protocol)


class _CustomICRS(coord.ICRS):
    default_representation = coord.PhysicsSphericalRepresentation


@pytest.mark.parametrize(
    "frame",
    [
        coord.SkyOffsetFrame(origin=coord.ICRS(0 * u.deg, 0 * u.deg)),
        coord.SkyOffsetFrame(
            5 * u.deg, 10 * u.deg, origin=coord.Galactic(2 * u.deg, -3 * u.deg)
        ),
        coord.SkyOffsetFrame(
            5 * u.deg,
            10 * u.deg,
            10 * u.pc,
            origin=coord.Galactic(2 * u.deg, -3 * u.deg),
            representation_type=coord.PhysicsSphericalRepresentation,
        ),
        coord.SkyOffsetFrame(
            5 * u.deg,
            10 * u.deg,
            0 * u.pc,
            origin=_CustomICRS(2 * u.deg, 3 * u.deg, 1 * u.pc),
        ),
        coord.SkyOffsetFrame(origin=coord.ICRS(70 * u.deg, 20 * u.deg), polar=True),
        coord.SkyOffsetFrame(
            phi=30 * u.deg,
            theta=1 * u.deg,
            r=3 * u.kpc,
            origin=coord.Galactic(70 * u.deg, 20 * u.deg),
            rotation=35 * u.deg,
            polar=True,
        ),
    ],
)
def test_skyoffset_pickle(pickle_protocol, frame):  # noqa: F811
    """
    This is a regression test for issue #9249:
    https://github.com/astropy/astropy/issues/9249
    """
    check_pickling_recovery(frame, pickle_protocol)


def test_skyoffset_polar_array_pickle(pickle_protocol):  # noqa: F811
    frame = coord.SkyOffsetFrame(
        phi=[30, 60] * u.deg,
        theta=[1, 2] * u.deg,
        r=3 * u.kpc,
        origin=coord.Galactic(70 * u.deg, 20 * u.deg),
        rotation=35 * u.deg,
        polar=True,
    )
    recovered = pickle.loads(pickle.dumps(frame, protocol=pickle_protocol))
    assert type(recovered) is type(frame)
    assert recovered.default_representation is coord.PhysicsSphericalRepresentation
    assert recovered.rotation == frame.rotation
    assert recovered.origin == frame.origin
    np.testing.assert_array_equal(recovered.phi, frame.phi)
    np.testing.assert_array_equal(recovered.theta, frame.theta)
    np.testing.assert_array_equal(recovered.r, frame.r)
