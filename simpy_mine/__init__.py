"""
The ``simpy_mine`` module aggregates simpy_mine's most used components into a single
namespace. This is purely for convenience. You can of course also access
everything (and more!) via their actual submodules.

The following tables list all of the available components in this module.

{toc}

"""
from pkgutil import extend_path

from simpy_mine.core import Environment
from simpy_mine.rt import RealtimeEnvironment
from simpy_mine.events import Event, Timeout, Process, AllOf, AnyOf, Interrupt
from simpy_mine.resources.resource import (
    Resource, PriorityResource, PreemptiveResource)
from simpy_mine.resources.container import Container
from simpy_mine.resources.store import Store, FilterStore
from simpy_mine.util import test


def compile_toc(entries, section_marker='='):
    """Compiles a list of sections with objects into sphinx formatted
    autosummary directives."""
    toc = ''
    for section, objs in entries:
        toc += '\n\n%s\n%s\n\n' % (section, section_marker * len(section))
        toc += '.. autosummary::\n\n'
        for obj in objs:
            toc += '    ~%s.%s\n' % (obj.__module__, obj.__name__)
    return toc


toc = (
    ('Environments', (
        Environment, RealtimeEnvironment,
    )),
    ('Events', (
        Event, Timeout, Process, AllOf, AnyOf, Interrupt,
    )),
    ('Resources', (
        Resource, PriorityResource, PreemptiveResource, Container, Store,
        FilterStore,
    )),
    ('Miscellaneous', (
        test,
    )),
)

# Use the toc to keep the documentation and the implementation in sync.
__doc__ = __doc__.format(toc=compile_toc(toc))
__all__ = [obj.__name__ for section, objs in toc for obj in objs]

__path__ = extend_path(__path__, __name__)
__version__ = '3.0.7'
