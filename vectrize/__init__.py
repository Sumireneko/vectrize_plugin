from krita import DockWidgetFactory, DockWidgetFactoryBase
from .vectrize import Vectrize

DOCKER_ID = 'Vectrize'

dock_right = getattr(DockWidgetFactoryBase, "DockRight",
                     DockWidgetFactoryBase.DockPosition.DockRight)

dock = DockWidgetFactory(DOCKER_ID, dock_right, Vectrize)
Krita.instance().addDockWidgetFactory(dock)
