from __future__ import print_function

import reach.utils as rutils
import reach.pndrs as rpndrs


tgt_info = rutils.initialise_tgt_info(
    True,
    70,
    False
)


stars = [
    "HR_2342",
    "ksi_Gem",
    "HR_2391",
    "HR_2426"
]


for star in stars:

    matched_id = rpndrs.match_target_name(
        tgt_info,
        star,
        verbose=False
    )

    observed_name = rpndrs.get_observed_target_name(
        tgt_info,
        matched_id,
        star
    )

    print(
        "%-15s -> %-15s -> %s"
        % (
            star,
            matched_id,
            observed_name
        )
    )