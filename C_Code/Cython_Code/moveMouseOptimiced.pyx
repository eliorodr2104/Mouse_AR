import collections
import sys
import time
import platform
import functools
from contextlib import contextmanager
import _pyautogui_osx as platformModule

cdef float MINIMUM_DURATION = 0.1
cdef float MINIMUM_SLEEP = 0.05
FAILSAFE = False
cdef list FAILSAFE_POINTS = [(0, 0)]

Point = collections.namedtuple("Point", "x y")
Size = collections.namedtuple("Size", "width height")

cpdef int min(int a, int b):
  """
  Cython inline function to find the minimum of two integers.
  """
  return a if a < b else b

cpdef int max(int a, int b):
  """
  Cython inline function to find the maximum of two integers.
  """
  return a if a > b else b

class PyAutoGUIException(Exception):
    """
    PyAutoGUI code will raise this exception class for any invalid actions. If PyAutoGUI raises some other exception,
    you should assume that this is caused by a bug in PyAutoGUI itself. (Including a failure to catch potential
    exceptions raised by PyAutoGUI.)
    """

    pass


class FailSafeException(PyAutoGUIException):
    """
    This exception is raised by PyAutoGUI functions when the user puts the mouse cursor into one of the "failsafe
    points" (by default, one of the four corners of the primary monitor). This exception shouldn't be caught; it's
    meant to provide a way to terminate a misbehaving script.
    """

    pass


class ImageNotFoundException(PyAutoGUIException):
    """
    This exception is the PyAutoGUI version of PyScreeze's `ImageNotFoundException`, which is raised when a locate*()
    function call is unable to find an image.

    Ideally, `pyscreeze.ImageNotFoundException` should never be raised by PyAutoGUI.
    """

if sys.version_info[0] == 2 or sys.version_info[0:2] in ((3, 1), (3, 2)):
    # Python 2 and 3.1 and 3.2 uses collections.Sequence
    from collections import Sequence
else:
    # Python 3.3+ uses collections.abc.Sequence
    from collections.abc import Sequence

cpdef float linear(float n):
  """
  Cython optimized function for linear interpolation between 0.0 and 1.0.
  """
  if n < 0.0 or n > 1.0:
    raise PyAutoGUIException("Argument must be between 0.0 and 1.0.")
    
  return n


def raisePyAutoGUIImageNotFoundException(wrappedFunction):
    """
    A decorator that wraps PyScreeze locate*() functions so that the PyAutoGUI user sees them raise PyAutoGUI's
    ImageNotFoundException rather than PyScreeze's ImageNotFoundException. This is because PyScreeze should be
    invisible to PyAutoGUI users.
    """

    @functools.wraps(wrappedFunction)
    def wrapper(*args, **kwargs):
        try:
            return wrappedFunction(*args, **kwargs)
        except pyscreeze.ImageNotFoundException:
            raise ImageNotFoundException  # Raise PyAutoGUI's ImageNotFoundException.

    return wrapper

try:
    import pyscreeze
    from pyscreeze import center, pixel, pixelMatchesColor, screenshot

    # Change the locate*() functions so that they raise PyAutoGUI's ImageNotFoundException instead.
    @raisePyAutoGUIImageNotFoundException
    def locate(*args, **kwargs):
        return pyscreeze.locate(*args, **kwargs)

    locate.__doc__ = pyscreeze.locate.__doc__

    @raisePyAutoGUIImageNotFoundException
    def locateAll(*args, **kwargs):
        return pyscreeze.locateAll(*args, **kwargs)

    locateAll.__doc__ = pyscreeze.locateAll.__doc__

    @raisePyAutoGUIImageNotFoundException
    def locateAllOnScreen(*args, **kwargs):
        return pyscreeze.locateAllOnScreen(*args, **kwargs)

    locateAllOnScreen.__doc__ = pyscreeze.locateAllOnScreen.__doc__

    @raisePyAutoGUIImageNotFoundException
    def locateCenterOnScreen(*args, **kwargs):
        return pyscreeze.locateCenterOnScreen(*args, **kwargs)

    locateCenterOnScreen.__doc__ = pyscreeze.locateCenterOnScreen.__doc__

    @raisePyAutoGUIImageNotFoundException
    def locateOnScreen(*args, **kwargs):
        return pyscreeze.locateOnScreen(*args, **kwargs)

    locateOnScreen.__doc__ = pyscreeze.locateOnScreen.__doc__

    @raisePyAutoGUIImageNotFoundException
    def locateOnWindow(*args, **kwargs):
        return pyscreeze.locateOnWindow(*args, **kwargs)

    locateOnWindow.__doc__ = pyscreeze.locateOnWindow.__doc__


except ImportError:
    # If pyscreeze module is not found, screenshot-related features will simply not work.
    def _couldNotImportPyScreeze(*unused_args, **unsed_kwargs):
        """
        This function raises ``PyAutoGUIException``. It's used for the PyScreeze function names if the PyScreeze module
        failed to be imported.
        """
        raise PyAutoGUIException(
            "PyAutoGUI was unable to import pyscreeze. (This is likely because you're running a version of Python that Pillow (which pyscreeze depends on) doesn't support currently.) Please install this module to enable the function you tried to call."
        )

    center = _couldNotImportPyScreeze
    #grab = _couldNotImportPyScreeze  # grab() was removed, use screenshot() instead
    locate = _couldNotImportPyScreeze
    locateAll = _couldNotImportPyScreeze
    locateAllOnScreen = _couldNotImportPyScreeze
    locateCenterOnScreen = _couldNotImportPyScreeze
    locateOnScreen = _couldNotImportPyScreeze
    locateOnWindow = _couldNotImportPyScreeze
    pixel = _couldNotImportPyScreeze
    pixelMatchesColor = _couldNotImportPyScreeze
    screenshot = _couldNotImportPyScreeze

"""
# The platformModule is where we reference the platform-specific functions.
if sys.platform.startswith("java"):
    # from . import _pyautogui_java as platformModule
    raise NotImplementedError("Jython is not yet supported by PyAutoGUI.")
elif sys.platform == "darwin":
    from . import _pyautogui_osx as platformModule
elif sys.platform == "win32":
    from . import _pyautogui_win as platformModule
elif platform.system() == "Linux":
    from . import _pyautogui_x11 as platformModule
else:
    raise NotImplementedError("Your platform (%s) is not supported by PyAutoGUI." % (platform.system()))

"""

def moveTo(
    int x, 
    int y, 
    duration = 0.0, 
    tween = linear, 
    logScreenshot = False, 
    _pause = True ):

    x, y = _normalizeXYArgs(x, y)

    #_logScreenshot(logScreenshot, "moveTo", "%s,%s" % (x, y), folder=".")
    _mouseMoveDrag(x, y, 0, 0, duration, tween)

def _mouseMoveDrag(int x, int y, int xOffset, int yOffset, float duration, tween=linear):
    cdef int xOptimized = x if x is not None else position()[0]  # Use x from position() if not provided
    cdef int yOptimized = y if y is not None else position()[1]
    cdef int xOffsetOptimized = int(xOffsetOptimized) if xOffsetOptimized is not None else 0
    cdef int yOffsetOptimized = int(yOffsetOptimized) if yOffsetOptimized is not None else 0
    cdef float durationOptimized = duration
    cdef int num_steps = 0
    cdef double sleep_amount = 0.0

    # Early return for no movement (avoid unnecessary calculations)
    if xOptimized is None and yOptimized is None and xOffsetOptimized == 0 and yOffsetOptimized == 0:
        return

    xOptimized += xOffsetOptimized
    yOptimized += yOffsetOptimized

    cdef int width, height
    width, height = size()

    # Clamp to screen bounds (assuming max/min behavior)
    xOptimized = max(0, min(xOptimized, width - 1))
    yOptimized = max(0, min(yOptimized, height - 1))

    cdef list steps

    # Instant movement for very short durations
    if durationOptimized <= MINIMUM_DURATION:
        steps = [(xOptimized, yOptimized)]
    else:
        # Calculate steps for smooth tweening
        num_steps = max(width, height)
        sleep_amount = durationOptimized / num_steps

        # Adjust num_steps and sleep_amount for minimum sleep time
        if sleep_amount < MINIMUM_SLEEP:
            num_steps = int(durationOptimized / MINIMUM_SLEEP)
            sleep_amount = durationOptimized / num_steps

        steps = [getPointOnLine(position()[0], position()[1], xOptimized, yOptimized, tween(n / num_steps)) for n in range(num_steps)]
        steps.append((xOptimized, yOptimized))  # Ensure final destination

    # Move the mouse along calculated steps
    for tweenX, tweenY in steps:
        if len(steps) > 1:
            time.sleep(sleep_amount)

        # Rounding and fail-safe check (assuming platform module handles clamping)
        platformModule._moveTo(int(round(tweenX)), int(round(tweenY)))
        if (tweenX, tweenY) not in FAILSAFE_POINTS:
            failSafeCheck()

    # Final fail-safe check after all steps (optional, depending on behavior)
    if (xOptimized, yOptimized) not in FAILSAFE_POINTS:
        failSafeCheck()

def _normalizeXYArgs(firstArg, secondArg):
    """
    Returns a ``Point`` object based on ``firstArg`` and ``secondArg``, which are the first two arguments passed to
    several PyAutoGUI functions. If ``firstArg`` and ``secondArg`` are both ``None``, returns the current mouse cursor
    position.

    ``firstArg`` and ``secondArg`` can be integers, a sequence of integers, or a string representing an image filename
    to find on the screen (and return the center coordinates of).
    """
    if firstArg is None and secondArg is None:
        return position()

    elif firstArg is None and secondArg is not None:
        return Point(int(position()[0]), int(secondArg))

    elif secondArg is None and firstArg is not None and not isinstance(firstArg, Sequence):
        return Point(int(firstArg), int(position()[1]))

    elif isinstance(firstArg, str):
        # If x is a string, we assume it's an image filename to locate on the screen:
        try:
            location = locateOnScreen(firstArg)
            # The following code only runs if pyscreeze.USE_IMAGE_NOT_FOUND_EXCEPTION is not set to True, meaning that
            # locateOnScreen() returns None if the image can't be found.
            if location is not None:
                return center(location)
            else:
                return None
        except pyscreeze.ImageNotFoundException:
            raise ImageNotFoundException

        return center(locateOnScreen(firstArg))

    elif isinstance(firstArg, Sequence):
        if len(firstArg) == 2:
            # firstArg is a two-integer tuple: (x, y)
            if secondArg is None:
                return Point(int(firstArg[0]), int(firstArg[1]))
            else:
                raise PyAutoGUIException(
                    "When passing a sequence for firstArg, secondArg must not be passed (received {0}).".format(
                        repr(secondArg)
                    )
                )
        elif len(firstArg) == 4:
            # firstArg is a four-integer tuple, (left, top, width, height), we should return the center point
            if secondArg is None:
                return center(firstArg)
            else:
                raise PyAutoGUIException(
                    "When passing a sequence for firstArg, secondArg must not be passed and default to None (received {0}).".format(
                        repr(secondArg)
                    )
                )
        else:
            raise PyAutoGUIException(
                "The supplied sequence must have exactly 2 or exactly 4 elements ({0} were received).".format(
                    len(firstArg)
                )
            )
    else:
        return Point(int(firstArg), int(secondArg))  # firstArg and secondArg are just x and y number values

def position(x = None, y = None):
    """
    Returns the current xy coordinates of the mouse cursor as a two-integer tuple.

    Args:
      x (int, None, optional) - If not None, this argument overrides the x in
        the return value.
      y (int, None, optional) - If not None, this argument overrides the y in
        the return value.

    Returns:
      (x, y) tuple of the current xy coordinates of the mouse cursor.

    NOTE: The position() function doesn't check for failsafe.
    """

    cdef int posx, posy
    posx, posy = platformModule._position()
    
    if x is not None:  # If set, the x parameter overrides the return value.
        posx = x

    if y is not None:  # If set, the y parameter overrides the return value.
        posy = y

    return Point(posx, posy)

def size():
    """Returns the width and height of the screen as a two-integer tuple.

    Returns:
      (width, height) tuple of the screen size, in pixels.
    """
    return Size(*platformModule._size())

def getPointOnLine(int x1, int y1, int x2, int y2, float n):
    """
    Restituisce una tupla (x, y) del punto che ha percorso una proporzione ``n`` lungo la linea definita dalle due coordinate
    ``x1``, ``y1`` e ``x2``, ``y2``.

    Questa funzione è stata copiata dal modulo pytweening, in modo che possa essere chiamata anche se PyTweening non è installato.
    """
    cdef int dx = x2 - x1
    cdef int dy = y2 - y1

    cdef float x = n * dx + x1
    cdef float y = n * dy + y1

    if n == 0.0:
        return x1, y1
    elif n == 1.0:
        return x2, y2

    return int(x), int(y)

def failSafeCheck():
    if FAILSAFE and tuple(position()) in FAILSAFE_POINTS:
        raise FailSafeException(
            "PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED."
        )