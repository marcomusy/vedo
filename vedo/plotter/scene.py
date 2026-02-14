"""Scene operations delegated from Plotter."""

from typing import Any

import vedo
import vedo.vtkclasses as vtki
from vedo import addons, utils
from vedo.plotter.events import Event


__docformat__ = "google"


def add(plotter, *objs, at=None) -> Any:
    """
    Append the input objects to the internal list of objects to be shown.

    Arguments:
        at : (int)
            add the object at the specified renderer
    """
    ren = plotter.renderer if at is None else plotter.renderers[at]

    objs = utils.flatten(objs)
    for ob in objs:
        if ob and ob not in plotter.objects:
            plotter.objects.append(ob)

    acts = plotter._scan_input_return_acts(objs)

    for a in acts:

        if ren:
            if isinstance(a, vedo.addons.BaseCutter):
                a.add_to(plotter)  # from cutters
                continue

            if isinstance(a, vtki.vtkLight):
                ren.AddLight(a)
                continue

            try:
                ren.AddActor(a)
            except TypeError:
                ren.AddActor(a.actor)

            try:
                ir = plotter.renderers.index(ren)
                a.rendered_at.add(ir)  # might not have rendered_at
            except (AttributeError, ValueError):
                pass

            if isinstance(a, vtki.vtkFollower):
                a.SetCamera(plotter.camera)
            elif isinstance(a, vedo.visual.LightKit):
                a.lightkit.AddLightsToRenderer(ren)

    return plotter

def remove(plotter, *objs, at=None) -> Any:
    """
    Remove input object to the internal list of objects to be shown.

    Objects to be removed can be referenced by their assigned name,
    or by passing the object instance itplotter.

    Wildcards are supported in the names.
    E.g. `Eleph*nt` or `Eleph?nt` or `Eleph[aio]nt` will match `Elephant`.

    Arguments:
        at : (int)
            remove the object at the specified renderer
    """
    ren = plotter.renderer if at is None else plotter.renderers[at]
    if not ren:
        return plotter
    ir = plotter.renderers.index(ren)

    on_scene_actors = plotter.get_actors(include_non_pickables=True)
    # print("remove() called", [objs])
    # print("on_scene_actors", (on_scene_actors))

    # add to objs_to_remove the ones with string name and remove the rest
    objs_to_remove = []
    for ob in utils.flatten(objs):
        if not ob:
            continue
        if isinstance(ob, str):
            name = ob
            for a in on_scene_actors:
                # print("->> checking", [a])
                try:
                    vobj = a.retrieve_object()
                    # print(" ->> found", [vobj], vobj.name)
                    if utils.string_match(name, vobj.name):
                        objs_to_remove.append(a)
                except AttributeError:
                    pass
        
        elif isinstance(ob, vedo.addons.BaseCutter):
            ob.remove_from(plotter)  # from cutters
            continue

        elif isinstance(ob, vedo.visual.LightKit):
            ob.lightkit.RemoveLightsFromRenderer(ren)

        else:
            objs_to_remove.append(ob)

    # remove objs_to_remove actors from the scene
    for ob in objs_to_remove:
        # print("->> removing", [ob])

        if hasattr(ob, "rendered_at"):
            ob.rendered_at.discard(ir)

        try: # vtk actor
            ren.RemoveActor(ob)
        except TypeError:
            try: # vedo object
                ren.RemoveActor(ob.actor)
                if hasattr(ob, "scalarbar") and ob.scalarbar:
                    ren.RemoveActor(ob.scalarbar)
                if hasattr(ob, "_caption") and ob._caption:
                    ren.RemoveActor(ob._caption)
                if hasattr(ob, "shadows") and ob.shadows:
                    for sha in ob.shadows: ren.RemoveActor(sha.actor)
                if hasattr(ob, "trail") and ob.trail:
                    ren.RemoveActor(ob.trail.actor)
                    ob.trail_points = []
                    if hasattr(ob.trail, "shadows") and ob.trail.shadows:
                        for sha in ob.trail.shadows: ren.RemoveActor(sha.actor)
            except AttributeError:
                pass

    plotter.objects = [ele for ele in plotter.objects if ele not in objs_to_remove]
    return plotter

def actors(plotter):
    """Return the list of actors."""
    return [ob.actor for ob in plotter.objects if hasattr(ob, "actor")]

def remove_lights(plotter) -> Any:
    """Remove all the present lights in the current renderer."""
    if plotter.renderer:
        plotter.renderer.RemoveAllLights()
    return plotter

def pop(plotter, at=None) -> Any:
    """
    Remove the last added object from the rendering window.
    This method is typically used in loops or callback functions.
    """
    if at is not None and not isinstance(at, int):
        # wrong usage pitfall
        vedo.logger.error("argument of pop() must be an integer")
        raise RuntimeError()

    if plotter.objects:
        plotter.remove(plotter.objects[-1], at)
    return plotter

def get_meshes(plotter, at=None, include_non_pickables=False, unpack_assemblies=True) -> list:
    """
    Return a list of Meshes from the specified renderer.

    Arguments:
        at : (int)
            specify which renderer to look at.
        include_non_pickables : (bool)
            include non-pickable objects
        unpack_assemblies : (bool)
            unpack assemblies into their components
    """
    if at is None:
        renderer = plotter.renderer
        at = plotter.renderers.index(renderer)
    elif isinstance(at, int):
        renderer = plotter.renderers[at]

    has_global_axes = False
    if isinstance(plotter.axes_instances[at], vedo.Assembly):
        has_global_axes = True

    if unpack_assemblies:
        acs = renderer.GetActors()
    else:
        acs = renderer.GetViewProps()

    objs = []
    acs.InitTraversal()
    for _ in range(acs.GetNumberOfItems()):

        if unpack_assemblies:
            a = acs.GetNextItem()
        else:
            a = acs.GetNextProp()

        if isinstance(a, vtki.vtkVolume):
            continue

        if include_non_pickables or a.GetPickable():
            if a == plotter.axes_instances[at]:
                continue
            if has_global_axes and a in plotter.axes_instances[at].actors:
                continue
            try:
                objs.append(a.retrieve_object())
            except AttributeError:
                pass
    return objs

def get_volumes(plotter, at=None, include_non_pickables=False) -> list:
    """
    Return a list of Volumes from the specified renderer.

    Arguments:
        at : (int)
            specify which renderer to look at
        include_non_pickables : (bool)
            include non-pickable objects
    """
    renderer = plotter.renderer if at is None else plotter.renderers[at]

    vols = []
    acs = renderer.GetVolumes()
    acs.InitTraversal()
    for _ in range(acs.GetNumberOfItems()):
        a = acs.GetNextItem()
        if include_non_pickables or a.GetPickable():
            try:
                vols.append(a.retrieve_object())
            except AttributeError:
                pass
    return vols

def get_actors(plotter, at=None, include_non_pickables=False) -> list:
    """
    Return a list of Volumes from the specified renderer.

    Arguments:
        at : (int)
            specify which renderer to look at
        include_non_pickables : (bool)
            include non-pickable objects
    """
    renderer = plotter.renderer if at is None else plotter.renderers[at]
    if renderer is None:
        return []

    acts = []
    acs = renderer.GetViewProps()
    acs.InitTraversal()
    for _ in range(acs.GetNumberOfItems()):
        a = acs.GetNextProp()
        if include_non_pickables or a.GetPickable():
            acts.append(a)
    return acts
