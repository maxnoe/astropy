# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains a helper function to fill erfa.astrom struct and a
ScienceState, which allows to speed up coordinate transformations at the
expense of accuracy.
"""
import warnings

import numpy as np
import erfa

from ..time import Time
from ..utils.state import ScienceState
from .. import units as u
from .builtin_frames.utils import (
    get_jd12, get_cip, prepare_earth_position_vel, get_polar_motion, get_dut1utc,
    pav2pv
)

from ..utils.exceptions import AstropyWarning
from ..table import Table

__all__ = []


class ErfaAstrom:
    '''
    The default provider for astrometry values.
    A utility class to extract the necessary arguments for
    erfa functions from frame attributes, call the corresponding
    erfa functions and return the astrom object.
    '''

    @staticmethod
    def apci(frame_or_coord):
        '''
        Wrapper for ``erfa.apci``, used in conversions CIRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a CIRS frame is expected.
        '''
        jd1_tt, jd2_tt = get_jd12(frame_or_coord.obstime, 'tt')
        cip = get_cip(jd1_tt, jd2_tt)
        earth_pv, earth_heliocentric = prepare_earth_position_vel(frame_or_coord.obstime)
        return erfa.apci(jd1_tt, jd2_tt, earth_pv, earth_heliocentric, *cip)

    @staticmethod
    def apcs(frame_or_coord):
        '''
        Wrapper for ``erfa.apci``, used in conversions GCRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a GCRS frame is expected.
        '''
        jd1_tt, jd2_tt = get_jd12(frame_or_coord.obstime, 'tt')
        obs_pv = pav2pv(
            frame_or_coord.obsgeoloc.get_xyz(xyz_axis=-1).value,
            frame_or_coord.obsgeovel.get_xyz(xyz_axis=-1).value
        )
        earth_pv, earth_heliocentric = prepare_earth_position_vel(frame_or_coord.obstime)
        return erfa.apcs(jd1_tt, jd2_tt, obs_pv, earth_pv, earth_heliocentric)

    @staticmethod
    def apio13(frame_or_coord):
        '''
        Wrapper for ``erfa.apio13``, used in conversions AltAz <-> CIRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, an AltAz frame is expected.
        '''
        lon, lat, height = frame_or_coord.location.to_geodetic('WGS84')

        jd1_utc, jd2_utc = get_jd12(frame_or_coord.obstime, 'utc')
        dut1utc = get_dut1utc(frame_or_coord.obstime)

        return erfa.apio13(
            jd1_utc, jd2_utc, dut1utc,
            lon.to_value(u.radian),
            lat.to_value(u.radian),
            height.to_value(u.m),
            *get_polar_motion(frame_or_coord.obstime),
            # all below are already in correct units because they are QuantityFrameAttribues
            frame_or_coord.pressure.value,
            frame_or_coord.temperature.value,
            frame_or_coord.relative_humidity.value,
            frame_or_coord.obswl.value,
        )


class ErfaAstromTableInterpolator(ErfaAstrom):
    '''
    A provider for astrometry values that does not call erfa
    for each individual timestamp but interpolates linearly
    between support points from a prefilled table.

    For the interpolation, float64 MJD values are used, so time precision
    for the interpolation will be around a microsecond.

    This can dramatically speed up coordinate transformations,
    e.g. between CIRS and ICRS.

    The precision of the transformation will still be in the order of microseconds
    for reasonable values of time_resolution, e.g. ``300 * u.s``.

    Users should benchmark performance and accuracy with the default transformation
    for their specific use case and then choose a suitable ``time_resolution``
    from there.

    This class is intended be used together with the ``erfa_astrom`` science state,
    e.g. in a context manager like this

    Example
    -------
    >>> from astropy.coordinates import SkyCoord, CIRS
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import numpy as np
    '''

    def __init__(self):
        self._table = None

    @classmethod
    def for_support(cls, support):
        '''Create a new ErfaAstromTableInterpolator for given support obstimes

        Arguments
        ---------
        '''
        instance = cls()
        instance.fill(support)
        return instance

    @classmethod
    def from_table(cls, table):
        instance = cls()
        instance._table = table
        return instance

    def fill(self, support):
        data = {'mjd': support.tt.mjd}
        jd1_tt, jd2_tt = get_jd12(support, 'tt')

        # CIP
        data['cip_x'], data['cip_y'], data['cip_z'] = get_cip(jd1_tt, jd2_tt)

        earth_pv, earth_heliocentric = prepare_earth_position_vel(support)
        for i, dim in enumerate('xyz'):
            for key in 'pv':
                data[f'earth_{key}_{dim}'] = earth_pv[key][:, i]

            data[f'earth_heliocentric_{dim}'] = earth_heliocentric[:, i]

        self._table = Table(data)

    def _check(self):
        if self._table is None:
            raise ValueError(
                'No values have yet been stored in the table of this interpolator'
                ', you need to call ``ErfaAstromTableInterpolator.fill``'
                ' or create the instance from an existing table.'
            )

    def _prepare_earth_position_vel(self, obstime):
        self._check()
        # do interpolation
        earth_pv = np.empty(obstime.shape, dtype=erfa.dt_pv)
        earth_heliocentric = np.empty(obstime.shape + (3,))

        # convert to tt to make sure it's in the same scale as the support
        mjd = obstime.tt.mjd
        support_mjd = self._table['mjd']

        for dim, dim_key in enumerate('xyz'):
            for key in 'pv':
                earth_pv[key][..., dim] = np.interp(
                    mjd,
                    support_mjd,
                    self._table[f'earth_{key}_{dim_key}'],
                )
            earth_heliocentric[..., dim] = np.interp(
                mjd, support_mjd, self._table[f'earth_heliocentric_{dim_key}']
            )

        return earth_pv, earth_heliocentric

    def _get_cip(self, obstime):
        self._check()
        # convert to tt to make sure it's in the same scale as the support
        mjd = obstime.tt.mjd

        return tuple(
            np.interp(mjd, self._table['mjd'], self._table[f'cip_{dim}'])
            for dim in 'xyz'
        )

    def apci(self, frame_or_coord):
        '''
        Wrapper for ``erfa.apci``, used in conversions CIRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a CIRS frame is expected.
        '''
        self._check()
        obstime = frame_or_coord.obstime

        cip = self._get_cip(obstime)
        earth_pv, earth_heliocentric = self._prepare_earth_position_vel(obstime)

        jd1_tt, jd2_tt = get_jd12(obstime, 'tt')
        astrom = erfa.apci(jd1_tt, jd2_tt, earth_pv, earth_heliocentric, *cip)
        return astrom

    def apcs(self, frame_or_coord):
        self._check()
        '''
        Wrapper for ``erfa.apci``, used in conversions GCRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a GCRS frame is expected.
        '''
        obstime = frame_or_coord.obstime

        # get the position and velocity arrays for the observatory.  Need to
        # have xyz in last dimension, and pos/vel in one-but-last.
        earth_pv, earth_heliocentric = self._prepare_earth_position_vel(obstime)
        pv = pav2pv(
            frame_or_coord.obsgeoloc.get_xyz(xyz_axis=-1).value,
            frame_or_coord.obsgeovel.get_xyz(xyz_axis=-1).value
        )
        jd1_tt, jd2_tt = get_jd12(obstime, 'tt')
        return erfa.apcs(jd1_tt, jd2_tt, pv, earth_pv, earth_heliocentric)


class ErfaAstromInterpolator(ErfaAstromTableInterpolator):
    '''
    A provider for astrometry values that does not call erfa
    for each individual timestamp but interpolates linearly
    between support points.

    For the interpolation, float64 MJD values are used, so time precision
    for the interpolation will be around a microsecond.

    This can dramatically speed up coordinate transformations,
    e.g. between CIRS and ICRS,
    when obstime is an array of many values (factors of 10 to > 100 depending
    on the selected resolution, number of points and the time range of the values).

    The precision of the transformation will still be in the order of microseconds
    for reasonable values of time_resolution, e.g. ``300 * u.s``.

    Users should benchmark performance and accuracy with the default transformation
    for their specific use case and then choose a suitable ``time_resolution``
    from there.

    This class is intended be used together with the ``erfa_astrom`` science state,
    e.g. in a context manager like this

    Example
    -------
    >>> from astropy.coordinates import SkyCoord, CIRS
    >>> from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import numpy as np

    >>> obstime = Time.now() + np.linspace(0, 4, 1000) * u.hour
    >>> with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
    ...    cirs = SkyCoord.from_name('Crab').transform_to(CIRS(obstime=obstime))  # doctest: +REMOTE_DATA
    '''

    @u.quantity_input(time_resolution=u.day)
    def __init__(self, time_resolution):
        if time_resolution.to_value(u.us) < 10:
            warnings.warn(
                f'Using {self.__class__.__name__} with `time_resolution`'
                ' below 10 microseconds might lead to numerical inaccuracies'
                ' as the MJD-based interpolation is limited by floating point '
                ' precision to about a microsecond of precision',
                AstropyWarning
            )
        self.mjd_resolution = time_resolution.to_value(u.day)

    def _get_support_points(self, obstime):
        '''
        Calculate support points for the interpolation.

        We divide the MJD by the time resolution (as single float64 values),
        and calculate ceil and floor.
        Then we take the unique and sorted values and scale back to MJD.
        This will create a sparse support for non-regular input obstimes.
        '''
        mjd_scaled = np.ravel(obstime.mjd / self. mjd_resolution)

        # ndmin=1 needed in case only a single coordinate is transformed
        mjd_lower = np.array(np.floor(mjd_scaled), ndmin=1, copy=False)
        mjd_upper = np.array(np.ceil(mjd_scaled), ndmin=1, copy=False)

        # unique already does sorting
        mjd_u = np.unique(np.concatenate([mjd_lower, mjd_upper]))

        return Time(
            mjd_u * self.mjd_resolution,
            format='mjd',
            scale=obstime.scale,
        )

    def apci(self, frame_or_coord):
        '''
        Wrapper for ``erfa.apci``, used in conversions CIRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a CIRS frame is expected.
        '''
        obstime = frame_or_coord.obstime

        # For this, we don't want to interpolate single obstimes
        if obstime.size == 1:
            return ErfaAstrom.apci(frame_or_coord)

        support = self._get_support_points(obstime)
        self.fill(support)

        return super().apci(frame_or_coord)

    def apcs(self, frame_or_coord):
        '''
        Wrapper for ``erfa.apci``, used in conversions GCRS <-> ICRS

        Arguments
        ---------
        frame_or_coord: ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a GCRS frame is expected.
        '''
        obstime = frame_or_coord.obstime

        # For this, we don't want to interpolate single obstimes
        if obstime.size == 1:
            return super().apci(frame_or_coord)

        support = self._get_support_points(obstime)
        self.fill(support)

        return super().apcs(frame_or_coord)


class erfa_astrom(ScienceState):
    """
    ScienceState to select with astrom provider is used in
    coordinate transformations.
    """

    _value = ErfaAstrom()

    @classmethod
    def validate(cls, value):
        if not isinstance(value, ErfaAstrom):
            raise TypeError(f'Must be an instance of {ErfaAstrom!r}')
        return value
