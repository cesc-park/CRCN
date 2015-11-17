import logging
import argparse
import sys
import os
import gzip
import shlex
import subprocess
import multiprocessing
import traceback
import time
import itertools
from multiprocessing import Pool
from functools import partial

def make_grid(testgrid_path,trees):
    testgrid_base=os.path.dirname(testgrid_path)
    content_path=os.path.join(testgrid_base,'tmp.txt')
    if type(trees) is dict:
        content_path=os.path.join(testgrid_base,trees['key'])
        trees=trees['trees']
    f = open(content_path, 'w')
    if type(trees)==list:
        f.write('\n'.join(trees))
    else:
        f.write(trees)
    f.close()
    params = {'TestGrid': './TestGrid','content':content_path}
    cmd_line = '%(TestGrid)s %(content)s' % (params)
    cmd_args = shlex.split(cmd_line)
    proc = subprocess.Popen(cmd_args,cwd=testgrid_base, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if os.path.exists(content_path):
        os.remove(content_path)
    # before returning we replace trees of empty segments (marked as so) by empty trees
    return stdout

def parse(content, args):
    """
    Parse a number of segments.

    Arguments
    ---------
    mem: how much memory is available to the parser
    parser: path to jar
    models: path to jar
    grammar: path to gz
    threads: how many threads are available to the parser
    maxlength: segment maximum length (longer segments are skipped and the output is an empty parse tree: (())
    empty_seg: the token that marks an empty segment

    Returns
    -------
    list of parse trees (as strings)
    """
    params = {'mem': args['mem'],
            'parser': args['parser'],
            'models': args['models'],
            'grammar': args['grammar'],
            'threads': args['threads'],
            'maxlength': args['max_length'],
            }
    #print content
    cmd_line = 'java -mx%(mem)dg -cp "%(parser)s:%(models)s" edu.stanford.nlp.parser.lexparser.LexicalizedParser -nthreads %(threads)d -sentences newline -maxLength %(maxlength)d -outputFormat oneline %(grammar)s -' % (params)
    cmd_args = shlex.split(cmd_line)
    logging.debug('running: %s', cmd_line)
    proc = subprocess.Popen(cmd_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(content)
    #print stdout
    # before returning we replace trees of empty segments (marked as so) by empty trees
    output = [line.strip() for line in stdout.split('\n')]
    return [ptb if seg != '<EMPTY>' else '(())' for seg, ptb in itertools.izip(content.split('\n'), output)]

def wrap_parse(content, args):
    """
    Wraps a call to `parse` in a try/except block so that one can use a Pool
    and still get decent error messages.

    Arguments
    ---------
    content: segments are strings
    args: a namespace, see `parse`

    Returns
    -------
    parse trees and time to parse
    """
    if content.strip()=="" or content is None:
        return None
    try:
        trees = parse(content, args)
        if len(trees)!=0:
            return trees
        else:
            return None
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))

def wrap_grid(trees,testgrid_path):
    """
    Wraps a call to `parse` in a try/except block so that one can use a Pool
    and still get decent error messages.

    Arguments
    ---------
    content: segments are strings
    args: a namespace, see `parse`

    Returns
    -------
    parse trees and time to parse
    """
    if trees is None:
        return None

    try:
        grid = make_grid(testgrid_path,trees)
        return grid
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))
def get_parsed_trees_multi_documents(contents,args):
    # distributes the jobs
    pool = Pool(args['jobs'])
    logging.info('Distributing %d jobs to %d workers', len(contents), args['jobs'])
    trees_list = pool.map(partial(wrap_parse, args=args), contents)
    pool.terminate()
    return trees_list
def get_grids_multi_documents(testgrid_path,trees_list,jobs):
    # distributes the jobs
    pool = Pool(jobs)
    logging.info('Distributing %d jobs to %d workers', len(trees_list), jobs)
    gird_list = pool.map(partial(wrap_grid, testgrid_path=testgrid_path), trees_list)
    pool.close()
    pool.join()
    return gird_list
def get_grids_a_document(testgrid_path,trees):
    grid = wrap_grid(trees,testgrid_path)
    return grid
def get_parsed_trees_a_document(content,args):
    trees = wrap_parse(content, args)
    return trees
