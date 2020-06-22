#!/usr/bin/env python
# -*- coding: utf-8

import sys
import argparse

import anvio.db as db
import anvio.utils as utils


import anvio.terminal as terminal

from anvio.errors import ConfigError

current_version, next_version = [x[1:] for x in __name__.split('_to_')]

trna_taxonomy_table_name                = 'trna_taxonomy'
trna_taxonomy_table_structure           = ['gene_callers_id', 'amino_acid', 'anticodon', 'accession', 'percent_identity', 't_domain', "t_phylum", "t_class", "t_order", "t_family", "t_genus", "t_species"]
trna_taxonomy_table_types               = [    'numeric'    ,    'text'   ,    'text'  ,    'text'  ,       'text'      ,   'text'  ,   'text'  ,   'text' ,  'text'  ,   'text'  ,   'text' ,   'text'   ]

run = terminal.Run()
progress = terminal.Progress()

def migrate(db_path):
    if db_path is None:
        raise ConfigError("No database path is given.")

    utils.is_contigs_db(db_path)

    contigs_db = db.DB(db_path, None, ignore_version = True)
    if str(contigs_db.get_version()) != current_version:
        raise ConfigError("Version of this contigs database is not %s (hence, this script cannot really do anything)." % current_version)

    progress.new("Creating a new table for tRNA taxonomy")
    progress.update("...")

    # just to be on the safe side.
    try:
        contigs_db.drop_table(trna_taxonomy_table_name)
    except:
        pass

    try:
        contigs_db.remove_meta_key_value_pair('trna_taxonomy_was_run')
        contigs_db.remove_meta_key_value_pair('trna_taxonomy_database_version')
    except:
        pass

    contigs_db.set_meta_value('trna_taxonomy_was_run', False)
    contigs_db.set_meta_value('trna_taxonomy_database_version', None)
    contigs_db.create_table(trna_taxonomy_table_name, trna_taxonomy_table_structure, trna_taxonomy_table_types)

    progress.update("Updating version")
    contigs_db.remove_meta_key_value_pair('version')
    contigs_db.set_version(next_version)

    progress.update("Committing changes")
    contigs_db.disconnect()

    progress.end()
    run.info_single("The contigs database is now %s. This upgrade added an empty tRNA taxonomy table "
                    "in it. Probably you will never use it becasue who cares about tRNA taxonomy amirite? "
                    "Well, IT WILL FOREVER BE THERE ANYWAY. BOOM." % (next_version), nl_after=1, nl_before=1, mc='green')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple script to upgrade CONTIGS.db from version %s to version %s' % (current_version, next_version))
    parser.add_argument('contigs_db', metavar = 'CONTIGS_DB', help = 'Contigs database at version %s' % current_version)
    args, unknown = parser.parse_known_args()

    try:
        migrate(args.contigs_db)
    except ConfigError as e:
        print(e)
        sys.exit(-1)
