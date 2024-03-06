import argparse
from pprint import pprint

def PRSMModelArgs():
    parser = argparse.ArgumentParser(
        description='Optional command line arguments for PRSM_ADMM Model')
    parser.add_argument('--regularization_strength',
                        type=float,
                        default=None,
                        help='Nuclear norm penalty strength. Useless in cross-validation')
    parser.add_argument('--testing_set_ratio',
                        type=float,
                        default=None,
                        help='Test set split ratio')
    parser.add_argument('--use_full_dataset',
                        action="store_true",
                        help='My GPU memory is very rich and can lift full dataset')
    parser.add_argument('--num_subset_feature',
                        type=int,
                        default=10000,
                        help='Size of sub-dataset. Useless when --use_full_dataset is set')

    cmdargs, other = parser.parse_known_args()

    if cmdargs.use_full_dataset is True:
        cmdargs.num_subset_feature = None
    if cmdargs.regularization_strength is None:
        cmdargs.regularization_strength = 100
    if cmdargs.testing_set_ratio is None:
        cmdargs.testing_set_ratio = 0.136

    pprint(cmdargs)

    return cmdargs, other