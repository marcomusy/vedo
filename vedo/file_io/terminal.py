from __future__ import annotations

"""Terminal interaction helpers."""

from vedo import colors

__docformat__ = "google"


def ask(*question, **kwarg) -> str:
    """
    Ask a question from command line. Return the answer as a string.
    See function `colors.printc()` for the description of the keyword options.

    Arguments:
        options : (list)
            a python list of possible answers to choose from.
        default : (str)
            the default answer when just hitting return.

    Example:
    ```python
    import vedo
    res = vedo.ask("Continue?", options=['Y','n'], default='Y', c='g')
    print(res)
    ```
    """
    kwarg.update({"end": " "})
    if "invert" not in kwarg:
        kwarg.update({"invert": True})
    if "box" in kwarg:
        kwarg.update({"box": ""})

    options = kwarg.pop("options", [])
    default = kwarg.pop("default", "")
    if options:
        opt = "["
        for o in options:
            opt += o + "/"
        opt = opt[:-1] + "]"
        colors.printc(*question, opt, **kwarg)
    else:
        colors.printc(*question, **kwarg)

    try:
        resp = input()
    except Exception:
        resp = ""
        return resp

    if options:
        if resp not in options:
            if default and str(repr(resp)) == "''":
                return default
            colors.printc("Please choose one option in:", opt, italic=True, bold=False)
            kwarg["options"] = options
            return ask(*question, **kwarg)  # ask again
    return resp
