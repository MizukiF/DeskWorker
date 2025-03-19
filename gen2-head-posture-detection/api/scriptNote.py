from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import cv2
import depthai as dai
from tools import *

import time
msgs = dict()

def add_msg(msg, name, seq = None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    
    # Each seq number has it's own dict of msgs
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg
    

def get_msgs():
    global msgs
    seq_remove = [] 