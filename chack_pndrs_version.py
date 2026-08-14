from __future__ import print_function

import inspect
import reach.pndrs as rpndrs


print("\n" + "=" * 79)
print("PNDRS MODULE BEING USED")
print("=" * 79)

print(rpndrs.__file__)


print("\n" + "=" * 79)
print("save_nightly_ldd ACTUALLY BEING USED")
print("=" * 79)

print(
    inspect.getsource(
        rpndrs.save_nightly_ldd
    )
)