from __future__ import print_function

import os
import sys


PIPELINE = "pipeline_ksigem_tests.py"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(
            "Expected exactly one occurrence of %r, found %i"
            % (old, source.count(old))
        )
    return source.replace(old, new, 1)


def main():
    pipeline_path = os.path.abspath(PIPELINE)

    if not os.path.isfile(pipeline_path):
        raise RuntimeError(
            "Run this script from the REACH directory containing %s"
            % PIPELINE
        )

    with open(pipeline_path, "r") as handle:
        source = handle.read()

    # Activate REACH's native calibrator-as-science experiment.
    source = replace_once(
        source,
        "calibrate_calibrators = False",
        "calibrate_calibrators = True",
    )
    source = replace_once(
        source,
        "test_all_cals = False",
        "test_all_cals = True",
    )

    # Keep the diagnostic deterministic whenever these switches are present.
    if "do_random_ifg_sampling = True" in source:
        source = replace_once(
            source,
            "do_random_ifg_sampling = True",
            "do_random_ifg_sampling = False",
        )

    if "do_gaussian_diam_sampling = True" in source:
        source = replace_once(
            source,
            "do_gaussian_diam_sampling = True",
            "do_gaussian_diam_sampling = False",
        )

    # The underlying pipeline should start from the complete ksi Gem sequence.
    sys.argv = [pipeline_path, "ALL"]

    namespace = {
        "__name__": "__main__",
        "__file__": pipeline_path,
    }

    exec(compile(source, pipeline_path, "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
