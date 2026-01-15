#!/usr/bin/env python
"""Template echo utility."""

import argparse
import csv
import gzip
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Import Data Manager DmLog utility.
# Messages emitted using this result in Task Events.
from dm_job_utilities.dm_log import DmLog
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# import deepchem as dc
# from deepchem.data import NumpyDataset
# from deepchem.splits import ScaffoldSplitter


ID_COL_NAME = "ID"
SMILES_COL_NAME = "SMILES"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def expand_path(path):
    """
    Create any necessary directories to ensure that the file path is valid

    :param path: a filename or directory that might or not exist
    """
    head_tail = os.path.split(path)
    if head_tail[0]:
        if not os.path.isdir(head_tail[0]):
            os.makedirs(head_tail[0], exist_ok=True)


def read_delimiter(sep):
    if sep:
        if "tab" == sep:
            delimiter = "\t"
        elif "space" == sep:
            delimiter = None
        elif "comma" == sep:
            delimiter = ","
        elif "pipe" == sep:
            delimiter = "|"
        else:
            delimiter = sep
    else:
        delimiter = None
    return delimiter


class SmilesReader:

    def __init__(
        self, filename, read_header, delimiter, id_column, mol_column, recs_to_read
    ):
        self.delimiter = delimiter if delimiter is not None else " "
        if mol_column is None:
            if id_column == 0:
                self.mol_column = 1
            else:
                self.mol_column = 0
        else:
            self.mol_column = mol_column

        if id_column is None:
            self.id_column = None
        else:
            self.id_column = int(id_column)

        tmp_reader, file_reader = self.create_readers(filename)
        # read header line
        if read_header:
            tokens = next(tmp_reader)
            # tokens = self.tokenize(line)
            self.field_names = []
            for token in tokens:
                self.field_names.append(token.strip())
        else:
            self.field_names = []
            for i in range(
                0,
                max(
                    1,
                    self.mol_column + 1,
                    0 if self.id_column is None else self.id_column + 1,
                ),
            ):
                self.field_names.append(None)
            self.field_names[self.mol_column] = SMILES_COL_NAME
            if self.id_column is not None:
                self.field_names[self.id_column] = ID_COL_NAME

        max_num_tokens = 0
        for i in range(0, recs_to_read):
            line = next(tmp_reader, None)
            if not line:
                break
            else:
                num_tokens = len(line)
                if num_tokens > max_num_tokens:
                    max_num_tokens = num_tokens

        if max_num_tokens > len(self.field_names):
            for i in range(len(self.field_names), max_num_tokens):
                self.field_names.append("field" + str(i + 1))

        file_reader.close()

        # now create the read reader and discard the header
        self.reader, self.file = self.create_readers(filename)
        if read_header:
            line = next(self.reader)

    def create_readers(self, filename):
        if filename.endswith(".gz"):
            r = gzip.open(filename, "rt", encoding="utf-8")
            return csv.reader(r, delimiter=self.delimiter), r
        else:
            r = open(filename, "rt", encoding="utf-8")
            return csv.reader(r, delimiter=self.delimiter), r

    def get_mol_field_name(self):
        if self.field_names:
            return self.field_names[self.mol_column]
        else:
            return None

    def read(self):
        tokens = next(self.reader, None)
        if tokens:
            smi = tokens[self.mol_column]
            if self.id_column is not None:
                mol_id = tokens[self.id_column]
            else:
                mol_id = None

            mol = Chem.MolFromSmiles(smi)
            props = []

            for i, token in enumerate(tokens):
                token = token.strip()
                if not (i == self.mol_column or i == self.id_column):
                    props.append(token)
                    if mol:
                        if self.field_names and len(self.field_names) > i:
                            mol.SetProp(self.field_names[i], token)
                        else:
                            mol.SetProp("field" + str(i), token)

            t = (mol, smi, mol_id, props)
            return t
        else:
            return None

    def get_extra_field_names(self):
        if self.field_names:
            results = []
            for i, name in enumerate(self.field_names):
                if i != 0 and i != self.id_column:
                    results.append(name)
            return results
        else:
            return []

    def close(self):
        self.file.close()


def get_scaffold(mol: Chem.rdchem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def group_by_scaffold(mol_list):
    scaffold_to_indices = defaultdict(list)

    for idx, mol in enumerate(mol_list):
        scaffold = get_scaffold(mol)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)

    return scaffold_to_indices


def weighted_scaffold_split(mol_list, split_fractions, seed=42):
    rng = np.random.default_rng(seed)

    scaffold_groups = group_by_scaffold(mol_list)

    # sort scaffold groups largest first
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    split_names = list(split_fractions.keys())
    n_total = len(mol_list)

    target_sizes = {
        name: int(round(frac * n_total)) for name, frac in split_fractions.items()
    }

    splits = {name: [] for name in split_names}
    current_sizes = {name: 0 for name in split_names}

    for group in groups:
        deficits = {
            name: target_sizes[name] - current_sizes[name] for name in split_names
        }

        viable = [name for name in split_names if deficits[name] >= len(group)]

        if viable:
            max_deficit = max(deficits[n] for n in viable)
            candidates = [n for n in viable if deficits[n] == max_deficit]
        else:
            min_size = min(current_sizes.values())
            candidates = [n for n in split_names if current_sizes[n] == min_size]

        target = rng.choice(candidates)

        splits[target].extend(group)
        current_sizes[target] += len(group)

    # TODO: error handling
    check_scaffold_leakage(mol_list, splits)

    return splits


def weighted_random_split(smiles_list, split_fractions, seed=42):
    """
    Returns indices for each split.
    """

    rng = np.random.default_rng(seed)
    n_samples = len(smiles_list)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    splits = {}
    start = 0

    for name, frac in split_fractions.items():
        size = int(round(frac * n_samples))
        splits[name] = indices[start : start + size].tolist()
        start += size

    # assign leftovers (rounding)
    leftovers = indices[start:]
    for i, idx in enumerate(leftovers):
        splits[list(splits.keys())[i % len(splits)]].append(idx)

    return splits


def check_scaffold_leakage(mols, splits):
    split_scaffolds = []

    for idxs in splits:
        s = set(get_scaffold(mols[i]) for i in idxs)
        split_scaffolds.append(s)

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_scaffolds[i] & split_scaffolds[j]
            assert not overlap, f"Leakage between split {i} and {j}"


def fragment(mol, fragment_method, *args, **kwargs):  # pylint: disable=unused-argument
    """
    Generate the largest fragment in the molecule e.g. typically a desalt operation

    To be used as a Pandas data frame vector function
    :param mol: The molecule to fragment
    :param mode: The strategy for picking the largest (mw or hac)
    :return:
    """

    frags = Chem.GetMolFrags(mol, asMols=True)

    if len(frags) == 1:
        return mol
    else:
        # TODO - handle ties
        biggest_mol = frags[0]
        if fragment_method == "hac":
            biggest_count = 0
            for frag in frags:
                hac = frag.GetNumHeavyAtoms()
                if hac > biggest_count:
                    biggest_count = hac
                    biggest_mol = frag
        elif fragment_method == "mw":
            biggest_mw = 0
            for frag in frags:
                mw = Descriptors.MolWt(frag)
                if mw > biggest_mw:
                    biggest_mw = mw
                    biggest_mol = frag
        else:
            DmLog.emit_event(f"Invalid fragment mode: {fragment_method}")
            raise ValueError("Invalid fragment mode:", fragment_method)

    return biggest_mol


def run(
    filename,
    delimiter=None,
    id_column=None,
    mol_column=None,
    y_column=None,
    omit_fields=False,
    read_header=False,
    write_header=False,
    fragment_method="hac",
    # missing_val=None,
    split_method="random",
    split_ratios=(),
    split_names=(),
):

    DmLog.emit_event("Splitter job started")

    from pathlib import Path

    DmLog.emit_event(f"cwd: {Path('.').absolute()}")

    SPLIT_METHODS = {
        "random": weighted_random_split,
        "scaffold": weighted_scaffold_split,
    }

    columns = list(set([id_column, mol_column, y_column]))

    df = pd.read_csv(
        filename,
        delimiter=delimiter,
        header=0 if read_header else None,
    )

    # need mol in several places
    df["rdkit_mol"] = df[mol_column].apply(
        lambda smiles: Chem.MolFromSmiles(smiles),  # pylint: disable=unnecessary-lambda
    )
    df["fragment"] = df["rdkit_mol"].apply(fragment, args=(fragment_method), axis=1)

    seed = 42

    if len(split_names) != len(split_ratios):
        split_names = [f"group_{i + 1}" for i in range(len(split_ratios))]

    split_groups = {split_names[i]: k for i, k in enumerate(split_ratios)}

    method = SPLIT_METHODS.get(split_method, weighted_random_split)

    splits = method(df["rdkit_mol"], split_fractions=split_groups, seed=seed)
    df = df.drop(["rdkit_mol", "fragment"], axis=1)

    # write groups to files
    DmLog.emit_event("Split finished, writing out set files")
    for k, v in splits.items():
        fname = f"{k}.smi"

        if omit_fields:
            df = df.loc[:, columns]

        df.iloc[v].to_csv(fname, encoding="utf-8", index=False, header=write_header)
        os.chmod(fname, 0o664)


def list_of_strings(arg):
    return arg.split(",")


def list_of_floats(arg):
    l = list(map(float, arg.split(",")))
    if not abs(sum(l) - 1.0) < 1e-6:
        DmLog.emit_event("The sum of splits must be equal to 1")
        raise argparse.ArgumentError(arg, "The sum of splits must be equal to 1")
    return l


def main():
    parser = argparse.ArgumentParser(description="Split dataset")
    input_group = parser.add_argument_group("Input/output options")
    input_group.add_argument(
        "-i", "--input", required=True, help="Input file (.smi or .sdf)"
    )
    input_group.add_argument(
        "--omit-fields",
        action="store_true",
        help="Don't include fields from the input in the output",
    )

    input_group.add_argument(
        "--id-column",
        help="Column for name field (zero based integer for .smi, text for SDF)",
    )
    input_group.add_argument(
        "--mol-column",
        type=str,
        help="Column name for molecule when using delineated text formats",
    )
    input_group.add_argument(
        "--y-column",
        help="Column name for the Y variable",
    )
    input_group.add_argument(
        "--read-header",
        action="store_true",
        help="Read a header line with the field names when reading .smi or .txt",
    )
    input_group.add_argument(
        "--write-header",
        action="store_true",
        help="Write a header line when writing .smi or .txt",
    )
    # to pass tab as the delimiter specify it as $'\t' or use one of
    # the symbolic names 'comma', 'tab', 'space' or 'pipe'
    input_group.add_argument("-d", "--delimiter", help="Delimiter when using SMILES")

    rdkit_generic_group = parser.add_argument_group("General RDKit options")
    rdkit_generic_group.add_argument(
        "--fragment-method",
        choices=["hac", "mw", "none"],
        default="hac",
        help="Strategy for picking largest fragment (mw or hac or none",
    )
    split_group = parser.add_argument_group("Splitting options")
    split_group.add_argument(
        "--split-method",
        choices=["random", "scaffold"],
        default="random",
    )
    split_group.add_argument(
        "--split-ratios",
        type=list_of_floats,
        default=[0.5, 0.25, 0.25],
    )
    split_group.add_argument(
        "--split-names",
        type=list_of_strings,
        default=["training", "test", "validation"],
    )

    args = parser.parse_args()
    delimiter = read_delimiter(args.delimiter)

    print(args)

    run(
        args.input,
        omit_fields=args.omit_fields,
        delimiter=delimiter,
        id_column=args.id_column,
        mol_column=args.mol_column,
        y_column=args.y_column,
        read_header=args.read_header,
        write_header=args.write_header,
        fragment_method=args.fragment_method,
        split_method=args.split_method,
        split_ratios=args.split_ratios,
        split_names=args.split_names,
    )


if __name__ == "__main__":
    # python src/train_test_split.py --input=data/caco2_mordred_filtered_scaled.smi --delimiter=comma --id-column=0 --mol-column=Drug --y-column=Y --read-header --write-header --split-method=random

    main()
