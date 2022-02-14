# Licensed under a 3-clause BSD style license - see LICENSE.rst

from functools import cache

from astropy import units as u
from astropy.coordinates.attributes import CoordinateAttribute, QuantityAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.representation import (
    PhysicsSphericalDifferential,
    PhysicsSphericalRepresentation,
)
from astropy.coordinates.transformations import (
    DynamicMatrixTransform,
    FunctionTransform,
)


@cache
def make_skyoffset_cls(framecls, polar=False):
    """
    Create a new class that is the sky offset frame for a specific class of
    origin frame and polar convention. If such a class has already been
    created, the same class will be returned.

    The default components are ``lon``/``lat`` for ordinary offset frames and
    ``phi``/``theta``/``r`` for polar offset frames.

    Parameters
    ----------
    framecls : `~astropy.coordinates.BaseCoordinateFrame` subclass
        The class to create the SkyOffsetFrame of.
    polar : bool, optional
        Whether to place the origin at the north pole instead of on the equator.

    Returns
    -------
    skyoffsetframecls : class
        The class for the new skyoffset frame.

    Notes
    -----
    This function is necessary because Astropy's frame transformations depend
    on connection between specific frame *classes*.  So each type of frame
    needs its own distinct skyoffset frame class.  This function generates
    just that class, as well as ensuring that only one example of such a class
    actually gets created in any given python session.
    """
    if polar:
        default_representation = PhysicsSphericalRepresentation
        default_differential = PhysicsSphericalDifferential
    else:
        default_representation = framecls._default_representation
        default_differential = framecls._default_differential

    # Create a new SkyOffsetFrame subclass for this frame class.
    name = ("SkyOffsetPolar" if polar else "SkyOffset") + framecls.__name__
    _SkyOffsetFramecls = type(
        name,
        (SkyOffsetFrame, framecls),
        {
            "origin": CoordinateAttribute(
                frame=framecls, default=None, doc="The origin of the offset frame"
            ),
            "_polar": polar,
            # The following two have to be done because otherwise we use the
            # defaults of SkyOffsetFrame set by BaseCoordinateFrame.
            "_default_representation": default_representation,
            "_default_differential": default_differential,
            "__doc__": SkyOffsetFrame.__doc__,
        },
    )

    @frame_transform_graph.transform(
        FunctionTransform, _SkyOffsetFramecls, _SkyOffsetFramecls
    )
    def skyoffset_to_skyoffset(from_skyoffset_coord, to_skyoffset_frame):
        """Transform between two skyoffset frames."""
        # This transform goes through the parent frames on each side.
        # from_frame -> from_frame.origin -> to_frame.origin -> to_frame
        tmp_from = from_skyoffset_coord.transform_to(from_skyoffset_coord.origin)
        tmp_to = tmp_from.transform_to(to_skyoffset_frame.origin)
        return tmp_to.transform_to(to_skyoffset_frame)

    @frame_transform_graph.transform(
        DynamicMatrixTransform, framecls, _SkyOffsetFramecls
    )
    def reference_to_skyoffset(reference_frame, skyoffset_frame):
        """Convert a reference coordinate to a sky offset frame."""
        # Define rotation matrices along the position angle vector, and
        # relative to the origin.
        origin = skyoffset_frame.origin.spherical
        matrix = (
            rotation_matrix(-skyoffset_frame.rotation, "x")
            @ rotation_matrix(-origin.lat, "y")
            @ rotation_matrix(origin.lon, "z")
        )
        if skyoffset_frame._polar:
            # The ordinary offset axes are (pointing, east, north). A cyclic
            # permutation gives (east, north, pointing), preserving handedness
            # and making phi increase from positive offset lon towards lat.
            matrix = matrix[..., [1, 2, 0], :]
        return matrix

    @frame_transform_graph.transform(
        DynamicMatrixTransform, _SkyOffsetFramecls, framecls
    )
    def skyoffset_to_reference(skyoffset_coord, reference_frame):
        """Convert a sky offset frame coordinate to the reference frame."""
        # use the forward transform, but just invert it
        R = reference_to_skyoffset(reference_frame, skyoffset_coord)
        # transpose is the inverse because R is a rotation matrix
        return matrix_transpose(R)

    return _SkyOffsetFramecls


class SkyOffsetFrame(BaseCoordinateFrame):
    """
    A frame which is relative to some specific position and oriented to match
    its frame.

    By default, SkyOffsetFrames have spherical component names of
    ``lon``/``lat``, rather than those of the frame of ``origin``. With
    ``polar=True``, the origin is at the north pole and the default components
    are ``phi``, ``theta``, and ``r``.

    This is useful for calculating offsets and dithers in the frame of the sky
    relative to an arbitrary position. Coordinates in this frame are both
    centered on the position specified by the ``origin`` coordinate, *and*
    they are oriented in the same manner as the
    ``origin`` frame.  E.g., if ``origin`` is `~astropy.coordinates.ICRS`, this
    object's ``lat`` will be pointed in the direction of Dec, while ``lon``
    will point in the direction of RA.

    For more on skyoffset frames, see :ref:`astropy:astropy-skyoffset-frames`.

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    origin : coordinate-like
        The coordinate which specifies the origin of this frame. Note that this
        origin is used purely for on-sky location/rotation.  It can have a
        ``distance`` but it will not be used by this ``SkyOffsetFrame``.
    rotation : angle-like
        The final rotation of the frame about the ``origin``. The sign of
        the rotation is the left-hand rule.  That is, an object at a
        particular position angle in the un-rotated system will be sent to
        the positive latitude (z) direction in the final frame. For polar
        frames, a positive rotation increases ``phi`` by the same angle.
    polar : bool, optional
        If `True`, place the origin at ``theta=0`` and use
        `~astropy.coordinates.PhysicsSphericalRepresentation`. The angle
        ``theta`` is the angular separation from the origin. The angle ``phi``
        increases from positive offset longitude towards positive offset
        latitude in the corresponding non-polar frame with the same rotation.
        At zero rotation, ``phi = 90 deg - position_angle``, modulo 360 degrees,
        where the position angle is measured in the frame of ``origin``.
        The default is `False`.


    Notes
    -----
    ``SkyOffsetFrame`` is a factory class.  That is, the objects that it
    yields are *not* actually objects of class ``SkyOffsetFrame``.  Instead,
    distinct classes are created on-the-fly for whatever the frame class is
    of ``origin`` and the value of ``polar``. The ``polar`` keyword selects
    the class; it is not a frame attribute and cannot be changed by replication.

    The polar angle ``phi`` is undefined at the origin and its antipode.
    In particular, position angles measured from the transformed origin at
    the pole depend on an arbitrary meridian; use the original frame to
    compare position angles.
    """

    rotation = QuantityAttribute(
        default=0, unit=u.deg, doc="The rotation angle for the frame orientation"
    )
    origin = CoordinateAttribute(
        default=None, frame=None, doc="The origin of the offset frame"
    )
    _polar = False

    def __new__(cls, *args, **kwargs):
        # We don't want to call this method if we've already set up
        # an skyoffset frame for this class.
        if not (issubclass(cls, SkyOffsetFrame) and cls is not SkyOffsetFrame):
            # We get the origin argument, and handle it here.
            try:
                origin_frame = kwargs["origin"]
            except KeyError:
                raise TypeError(
                    "Can't initialize a SkyOffsetFrame without origin= keyword."
                )
            if hasattr(origin_frame, "frame"):
                origin_frame = origin_frame.frame
            polar = kwargs.get("polar", False)
            newcls = make_skyoffset_cls(origin_frame.__class__, polar=polar)
            return newcls.__new__(newcls, *args, **kwargs)

        # http://stackoverflow.com/questions/19277399/why-does-object-new-work-differently-in-these-three-cases
        # See above for why this is necessary. Basically, because some child
        # may override __new__, we must override it here to never pass
        # arguments to the object.__new__ method.
        if super().__new__ is object.__new__:
            return super().__new__(cls)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # polar selects the generated class; it is not a frame attribute.
        polar = kwargs.pop("polar", self._polar)
        if polar != self._polar:
            raise ValueError(
                "The polar convention is fixed by the frame class. "
                "Create a new SkyOffsetFrame to change it."
            )
        super().__init__(*args, **kwargs)
        if self.origin is not None and not self.origin.has_data:
            raise ValueError("The origin supplied to SkyOffsetFrame has no data.")
        if self.has_data:
            self._set_skyoffset_data_lon_wrap_angle(self.data)

    @staticmethod
    def _set_skyoffset_data_lon_wrap_angle(data):
        if hasattr(data, "lon"):
            data.lon.wrap_angle = 180.0 * u.deg
        return data

    def represent_as(self, base, s="base", in_frame_units=False):
        """
        Ensure the wrap angle for any spherical
        representations.
        """
        data = super().represent_as(base, s, in_frame_units=in_frame_units)
        self._set_skyoffset_data_lon_wrap_angle(data)
        return data

    def __reduce__(self):
        return (_skyoffset_reducer, (self.origin, self._polar), self.__dict__)


def _skyoffset_reducer(origin, polar=False):
    return SkyOffsetFrame.__new__(SkyOffsetFrame, origin=origin, polar=polar)
